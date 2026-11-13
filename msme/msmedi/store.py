"""Per-user storage (SQLite). Every query is scoped by user_id and parameterised."""
import datetime as dt, hashlib, json, os, sqlite3, zlib
from contextlib import closing
import pandas as pd

def db_path():
    return os.environ.get("MSME_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "msmedi.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, name TEXT, salt BLOB, hash BLOB,
  iters INTEGER, created_at TEXT, failed INTEGER DEFAULT 0, locked_until REAL DEFAULT 0, settings TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS sales(user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, branch TEXT NOT NULL, day TEXT NOT NULL,
  amount REAL NOT NULL, PRIMARY KEY(user_id, branch, day));
CREATE TABLE IF NOT EXISTS branches(user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, branch TEXT NOT NULL,
  business TEXT NOT NULL DEFAULT 'My business', PRIMARY KEY(user_id, branch));
CREATE TABLE IF NOT EXISTS macro(user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, series TEXT NOT NULL, ref_month TEXT NOT NULL,
  value REAL, release_date TEXT, PRIMARY KEY(user_id, series, ref_month));
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT, fingerprint TEXT, results TEXT, preds BLOB);
CREATE TABLE IF NOT EXISTS forecast_log(run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE, user_id INTEGER NOT NULL,
  branch TEXT, horizon INTEGER, target_week TEXT, point REAL, lo REAL, hi REAL, model TEXT);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  branch TEXT, decision TEXT, note TEXT, created_at TEXT, latest_data_week TEXT);
"""

def _c():
    c = sqlite3.connect(db_path()); c.row_factory = sqlite3.Row; c.execute("PRAGMA foreign_keys=ON"); return c

def _run(sql, args=(), many=False):
    with closing(_c()) as c:
        cur = c.executemany(sql, args) if many else c.execute(sql, args); c.commit(); return cur.lastrowid

def _all(sql, args=()):
    with closing(_c()) as c: return [dict(r) for r in c.execute(sql, args).fetchall()]

def init():
    with closing(_c()) as c: c.executescript(SCHEMA); c.commit()

now = lambda: dt.datetime.now().isoformat(timespec="seconds")

# ---- users
def create_user(email, name, salt, h, iters): return _run("INSERT INTO users(email,name,salt,hash,iters,created_at) VALUES(?,?,?,?,?,?)", (email, name, salt, h, iters, now()))
def get_user_by_email(email): r = _all("SELECT * FROM users WHERE email=?", (email,)); return r[0] if r else None
def get_user(uid): r = _all("SELECT * FROM users WHERE id=?", (uid,)); return r[0] if r else None
def set_password(uid, salt, h, iters): _run("UPDATE users SET salt=?, hash=?, iters=? WHERE id=?", (salt, h, iters, uid))
def record_login(uid, ok, max_fails=5, lock_seconds=900):
    import time
    if ok: _run("UPDATE users SET failed=0, locked_until=0 WHERE id=?", (uid,)); return
    f = get_user(uid)["failed"] + 1
    _run("UPDATE users SET failed=?, locked_until=? WHERE id=?", (0 if f >= max_fails else f, time.time() + lock_seconds if f >= max_fails else 0, uid))
def get_settings(uid): return json.loads(get_user(uid)["settings"] or "{}")
def set_settings(uid, **kw): s = get_settings(uid); s.update(kw); _run("UPDATE users SET settings=? WHERE id=?", (json.dumps(s), uid))

# ---- sales and branches
def upsert_sales(uid, df):
    """df: biz, date, sales (daily or weekly). Returns (added, updated)."""
    n0 = _all("SELECT COUNT(*) n FROM sales WHERE user_id=?", (uid,))[0]["n"]
    rows = [(uid, str(b), pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for b, d, v in zip(df["biz"], df["date"], df["sales"])]
    _run("INSERT INTO sales(user_id,branch,day,amount) VALUES(?,?,?,?) ON CONFLICT(user_id,branch,day) DO UPDATE SET amount=excluded.amount", rows, many=True)
    n1 = _all("SELECT COUNT(*) n FROM sales WHERE user_id=?", (uid,))[0]["n"]
    _run("INSERT OR IGNORE INTO branches(user_id,branch) VALUES(?,?)", [(uid, b) for b in sorted(set(r[1] for r in rows))], many=True)
    return n1 - n0, len(rows) - (n1 - n0)

def get_sales(uid):
    r = _all("SELECT branch AS biz, day AS date, amount AS sales FROM sales WHERE user_id=?", (uid,))
    return pd.DataFrame(r, columns=["biz", "date", "sales"]) if r else pd.DataFrame(columns=["biz", "date", "sales"])

def get_branches(uid): return pd.DataFrame(_all("SELECT branch, business FROM branches WHERE user_id=? ORDER BY business, branch", (uid,)), columns=["branch", "business"])
def set_businesses(uid, mapping):
    _run("UPDATE branches SET business=? WHERE user_id=? AND branch=?", [((v or "My business").strip() or "My business", uid, k) for k, v in mapping.items()], many=True)

# ---- outside data
def upsert_macro(uid, m):
    rows = [(uid, str(r.series), pd.Timestamp(r.ref_month).strftime("%Y-%m-%d"), float(r.value) if r.value == r.value else None,
             pd.Timestamp(r.release_date).strftime("%Y-%m-%d")) for r in m.itertuples()]
    _run("INSERT INTO macro VALUES(?,?,?,?,?) ON CONFLICT(user_id,series,ref_month) DO UPDATE SET value=excluded.value, release_date=excluded.release_date", rows, many=True)
def get_macro(uid):
    r = _all("SELECT series, ref_month, value, release_date FROM macro WHERE user_id=?", (uid,))
    if not r: return None
    m = pd.DataFrame(r); m["ref_month"] = pd.to_datetime(m["ref_month"]); m["release_date"] = pd.to_datetime(m["release_date"]); return m

def fingerprint(uid, extra=""):
    h = hashlib.sha256(extra.encode())
    with closing(_c()) as c:
        for t, q in (("s", "SELECT branch,day,amount FROM sales WHERE user_id=? ORDER BY branch,day"),
                     ("m", "SELECT series,ref_month,value,release_date FROM macro WHERE user_id=? ORDER BY series,ref_month")):
            h.update(t.encode())
            for row in c.execute(q, (uid,)): h.update(repr(tuple(row)).encode())
    return h.hexdigest()

# ---- runs, forecast log, decisions
def save_run(uid, fp, results, preds, forecast_rows, keep=20):
    rid = _run("INSERT INTO runs(user_id,created_at,fingerprint,results,preds) VALUES(?,?,?,?,?)",
               (uid, now(), fp, json.dumps(results, default=str), zlib.compress(preds.to_csv(index=False).encode())))
    _run("INSERT INTO forecast_log VALUES(?,?,?,?,?,?,?,?,?)", [(rid, uid, *r) for r in forecast_rows], many=True)
    old = _all("SELECT id FROM runs WHERE user_id=? ORDER BY id DESC", (uid,))[keep:]
    for o in old: _run("DELETE FROM runs WHERE id=? AND user_id=?", (o["id"], uid))
    return rid

def latest_run(uid):
    r = _all("SELECT id, created_at, fingerprint, results, preds FROM runs WHERE user_id=? ORDER BY id DESC LIMIT 1", (uid,))
    if not r: return None
    import io
    r = r[0]; r["results"] = json.loads(r["results"]); r["preds"] = pd.read_csv(io.StringIO(zlib.decompress(r["preds"]).decode())); return r

def list_runs(uid): return pd.DataFrame(_all("SELECT id, created_at, substr(fingerprint,1,8) AS data_version FROM runs WHERE user_id=? ORDER BY id DESC", (uid,)))
def get_forecast_log(uid):
    return pd.DataFrame(_all("SELECT f.run_id, r.created_at, f.branch, f.horizon, f.target_week, f.point, f.lo, f.hi, f.model FROM forecast_log f JOIN runs r ON r.id=f.run_id WHERE f.user_id=? ORDER BY f.run_id", (uid,)))
def add_decision(uid, branch, decision, note, latest_week):
    _run("INSERT INTO decisions(user_id,branch,decision,note,created_at,latest_data_week) VALUES(?,?,?,?,?,?)", (uid, branch, decision, note, now(), latest_week))
def get_decisions(uid): return pd.DataFrame(_all("SELECT branch, decision, note, created_at, latest_data_week FROM decisions WHERE user_id=? ORDER BY id DESC", (uid,)))

# ---- erasure
def delete_user_data(uid):
    for t in ("forecast_log", "runs", "decisions", "macro", "branches", "sales"): _run(f"DELETE FROM {t} WHERE user_id=?", (uid,))
def delete_user(uid): delete_user_data(uid); _run("DELETE FROM users WHERE id=?", (uid,))
