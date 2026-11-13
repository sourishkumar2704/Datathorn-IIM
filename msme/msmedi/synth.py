"""Synthetic shops with a planted outside-signal effect (scale 0 = none). For pipeline tests only."""
import numpy as np, pandas as pd
from .features import event_features, INDIA_EVENTS, macro_asof, add_release_dates

SECT = {"leather": (0.10, 0.10, 1.0, 0.6), "grocery": (0.12, 0.06, 0.4, 1.0),
        "sweets": (0.45, 0.12, 0.8, 0.8), "textiles": (0.20, 0.14, 1.0, 0.5)}

def macro_frame():
    rng = np.random.default_rng(0)
    cpi_real = [101.67,101.32,101.39,101.58,101.9,102.51,103.35,103.74,103.74,103.74,104.01,104.1,104.45,104.57,104.84,105.12,105.91,107.0,107.95,108.74]
    wpi_real = [101,100.9,101.5,101.9,102.3,103.1,104.6,108.9,110.1,110.3,110.0,110.8,112.5]
    months = pd.period_range("2023-10", "2026-09", freq="M"); rows = []
    cpi = {p: v for p, v in zip(pd.period_range("2025-01", "2026-08", freq="M"), cpi_real)}
    wpi = {p: v for p, v in zip(pd.period_range("2025-09", "2026-09", freq="M"), wpi_real)}
    cpi[pd.Period("2026-09", freq="M")] = 108.74 * 1.007   # placeholder, dropped below (not released yet)
    for p in months[::-1]:
        if p not in cpi: cpi[p] = cpi[p + 1] / (1 + 0.0035 + rng.normal(0, 0.001))
        if p not in wpi: wpi[p] = wpi[p + 1] * (1 + rng.normal(0, 0.003))
    for p in months:
        rows.append(dict(series="cpi", ref_month=p.to_timestamp(), value=cpi[p]))
        rows.append(dict(series="wpi", ref_month=p.to_timestamp(), value=wpi[p]))
    m = pd.DataFrame(rows); return add_release_dates(m[~((m.series == "cpi") & (m.ref_month == "2026-09-01"))])

def generate(scale=0.0, seed=42, n_biz=24, n_weeks=156):
    rng = np.random.default_rng(seed); macro = macro_frame()
    weeks = pd.date_range("2023-10-02", periods=n_weeks, freq="7D")
    mac = [macro_asof(macro, w) for w in weeks]; names = list(SECT); cols = {}; sector = {}
    for b in range(n_biz):
        sec = names[b % 4]; fu, amp, sw, sc = SECT[sec]; sector[f"shop{b:02d}"] = sec
        base = np.exp(rng.normal(np.log(450000), 0.5)); g = rng.normal(0.08, 0.05); ph = rng.uniform(0, 6.28)
        sw *= rng.uniform(0.6, 1.4); sc *= rng.uniform(0.6, 1.4); u = 0.0; ys = []
        for t, w in enumerate(weeks):
            doy = w.dayofyear / 365.25 * 2 * np.pi; ev = event_features(w, {"diwali": INDIA_EVENTS["diwali"], "holi": INDIA_EVENTS["holi"]})
            fest = fu * (ev["diwali_k"] + 0.4 * ev["holi_k"]); m = mac[t]; eff = 0.0
            if scale and not np.isnan(m.get("wpi_3m", np.nan)) and not np.isnan(m.get("cpi_3m", np.nan)):
                eff = -scale * (0.005 * sw * m["wpi_3m"] + 0.010 * sc * m["cpi_3m"])
            u = 0.5 * u + rng.normal(0, 0.08); shock = rng.normal(0, 0.25) if rng.random() < 0.01 else 0.0
            ys.append(base * np.exp(g * t / 52 + amp * np.sin(doy + ph) + 0.3 * amp * np.cos(2 * doy) + fest + eff + u + shock))
        cols[f"shop{b:02d}"] = ys
    return pd.DataFrame(cols, index=weeks), macro, sector
