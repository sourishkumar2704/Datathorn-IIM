"""Calendar and outside-signal features. Outside signals are vintage-aware: a value is only
visible at time `asof` if its release date is on or before `asof`."""
import numpy as np, pandas as pd

# Verify festival dates against an official calendar before relying on them.
INDIA_EVENTS = {
    "diwali": ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29"],
    "holi": ["2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22"],
    "dussehra": ["2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20"],
    "eid_al_fitr": ["2023-04-22", "2024-04-11", "2025-03-31", "2026-03-20"],
    "raksha_bandhan": ["2023-08-30", "2024-08-19", "2025-08-09", "2026-08-28"],
}
UK_EVENTS = {"christmas": ["2009-12-25", "2010-12-25", "2011-12-25", "2012-12-25"]}
EVENT_SETS = {"india": INDIA_EVENTS, "uk": UK_EVENTS, "none": {}}
EVENT_WIDTH = {"christmas": 6.0}   # wider build-up for gift/seasonal peaks; default 3 weeks
DEFAULT_RELEASE_DAY = {"cpi": 12, "wpi": 14}   # day of the following month

def _dates(v): return [pd.Timestamp(x) for x in v]

def event_features(week_start, events):
    f = {}
    for name, ds in events.items():
        ds = _dates(ds)
        signed = [(d - week_start).days / 7.0 for d in ds]
        nearest = min(signed, key=abs)
        w = EVENT_WIDTH.get(name, 3.0)
        f[f"{name}_k"] = float(np.exp(-0.5 * (nearest / (w / 2)) ** 2))
        f[f"{name}_wk"] = float(np.clip(nearest, -12, 12))
    return f

def add_release_dates(macro, default_day=15):
    """macro columns: series, ref_month, value, [release_date]. Fill missing release dates."""
    m = macro.copy()
    m["ref_month"] = pd.to_datetime(m["ref_month"]).dt.to_period("M").dt.to_timestamp()
    if "release_date" not in m: m["release_date"] = pd.NaT
    m["release_date"] = pd.to_datetime(m["release_date"])
    fill = m.apply(lambda r: r["ref_month"] + pd.offsets.MonthBegin(1)
                   + pd.Timedelta(days=DEFAULT_RELEASE_DAY.get(str(r["series"]).lower(), default_day) - 1), axis=1)
    m["release_date"] = m["release_date"].fillna(fill)
    return m.sort_values(["series", "ref_month"]).reset_index(drop=True)

def macro_asof(macro, asof):
    """Latest visible level, 1-month and 3-month change per series, as known at `asof`."""
    f = {}
    if macro is None or len(macro) == 0: return f
    for s, g in macro.groupby("series"):
        v = g[g["release_date"] <= asof].dropna(subset=["value"]).sort_values("ref_month")
        vals = v["value"].values
        if len(vals) >= 4:
            f[f"{s}_3m"] = (vals[-1] / vals[-4] - 1) * 100
            f[f"{s}_1m"] = (vals[-1] / vals[-2] - 1) * 100
        else:
            f[f"{s}_3m"] = np.nan; f[f"{s}_1m"] = np.nan
    return f

def macro_fresh(macro, asof, max_age_days=75):
    """True if every series has a release visible at `asof` that is not stale (monthly data should be under ~75 days old)."""
    if macro is None or len(macro) == 0: return False
    for s, g in macro.groupby("series"):
        seen = g[g["release_date"] <= asof]["release_date"]
        if seen.empty or (asof - seen.max()).days > max_age_days: return False
    return True

EPS = 0.05
def to_log(y, med): return np.log(y / med + EPS)
def from_log(l, med): return np.maximum((np.exp(l) - EPS) * med, 0.0)

def build_rows(Y, h, events, macro=None, targets_known=True, pred_origin=None, sector=None):
    """Rows (shop, origin, target) for direct h-week-ahead forecasting.
    Y: DataFrame, index = Monday week starts, columns = shops, NaN where unknown."""
    weeks = Y.index; n = len(weeks); step = pd.Timedelta(days=7)
    med = Y.median(skipna=True).replace(0, np.nan).fillna(1.0)
    L = pd.DataFrame({c: to_log(Y[c].values, med[c]) for c in Y.columns}, index=weeks)
    ev_cache, mac_cache, out = {}, {}, []
    origins = [pred_origin] if pred_origin is not None else range(12, n - h)
    for o in origins:
        tgt = o + h; tw = weeks[0] + step * tgt
        if tgt not in ev_cache: ev_cache[tgt] = event_features(tw, events)
        if o not in mac_cache: mac_cache[o] = macro_asof(macro, weeks[0] + step * o + step) if macro is not None else {}
        doy = tw.dayofyear / 365.25 * 2 * np.pi
        for c in Y.columns:
            ly = L[c].values
            win = ly[o - 12:o + 1]
            if np.isnan(win).any(): continue
            m4 = ly[o - 3:o + 1].mean()
            f = dict(biz=c, origin=o, target=tgt, level=m4, med=med[c],
                     l0=ly[o] - m4, l1=ly[o - 1] - m4, l2=ly[o - 2] - m4, l3=ly[o - 3] - m4,
                     m13=win.mean() - m4, vol13=win.std(),
                     lag52=(ly[tgt - 52] - m4) if (tgt - 52 >= 0 and tgt - 52 < n and not np.isnan(ly[tgt - 52])) else np.nan,
                     sin1=np.sin(doy), cos1=np.cos(doy), sin2=np.sin(2 * doy), cos2=np.cos(2 * doy),
                     naive4=from_log(m4, med[c]))
            if sector is not None: f["sector"] = sector.get(c, -1)
            f.update(ev_cache[tgt]); f.update(mac_cache[o])
            if targets_known:
                if tgt >= n or np.isnan(ly[tgt]): continue
                f["y_true"] = Y[c].values[tgt]; f["resid"] = ly[tgt] - m4
                f["snaive"] = Y[c].values[tgt - 52] if tgt - 52 >= 0 else np.nan
            out.append(f)
    return pd.DataFrame(out)

META = {"biz", "origin", "target", "level", "med", "y_true", "resid", "naive4", "snaive"}
def feature_cols(D, with_macro):
    cols = [c for c in D.columns if c not in META]
    if not with_macro: cols = [c for c in cols if not (c.endswith("_3m") or c.endswith("_1m"))]
    return cols
