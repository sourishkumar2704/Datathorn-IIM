"""Accounts: salted PBKDF2 password hashing, lockout after repeated failures, generic error messages."""
import hashlib, hmac, os, re, time
from . import store

ITERS = 240_000
MAX_FAILS, LOCK_SECONDS = 5, 15 * 60
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _hash(pw, salt, iters=ITERS):
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, iters)

def check_password_strength(email, pw):
    if len(pw) < 8: return "Use at least 8 characters."
    if pw.lower() in (email.lower(), email.split("@")[0].lower()): return "Password must not be your email."
    if pw.isdigit() or pw.isalpha(): return "Mix letters with numbers or symbols."
    return None

def register(email, name, pw):
    """Returns (user_id, None) or (None, message)."""
    email = (email or "").strip().lower(); name = (name or "").strip() or email.split("@")[0]
    if not EMAIL.match(email): return None, "Enter a valid email address."
    bad = check_password_strength(email, pw or "")
    if bad: return None, bad
    if store.get_user_by_email(email): return None, "An account with this email already exists."
    salt = os.urandom(16)
    return store.create_user(email, name, salt, _hash(pw, salt), ITERS), None

def login(email, pw):
    """Returns (user dict, None) or (None, message). Same message for unknown email and wrong password."""
    email = (email or "").strip().lower(); u = store.get_user_by_email(email)
    generic = "Incorrect email or password."
    if u is None:
        _hash(pw or "", b"0" * 16)          # equalise timing so unknown emails are not revealed
        return None, generic
    if u["locked_until"] and u["locked_until"] > time.time():
        return None, f"Too many attempts. Try again in {int((u['locked_until'] - time.time()) // 60) + 1} minutes."
    if hmac.compare_digest(_hash(pw or "", u["salt"], u["iters"]), u["hash"]):
        store.record_login(u["id"], ok=True); return dict(u), None
    store.record_login(u["id"], ok=False, max_fails=MAX_FAILS, lock_seconds=LOCK_SECONDS)
    return None, generic

def change_password(uid, old, new):
    u = store.get_user(uid)
    if not hmac.compare_digest(_hash(old or "", u["salt"], u["iters"]), u["hash"]): return "Current password is incorrect."
    bad = check_password_strength(u["email"], new or "")
    if bad: return bad
    salt = os.urandom(16); store.set_password(uid, salt, _hash(new, salt), ITERS); return None
