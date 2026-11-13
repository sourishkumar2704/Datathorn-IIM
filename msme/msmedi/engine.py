"""Model ladder, rolling-origin backtest, calibrated intervals, signal verdict, future forecasts."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .features import build_rows, feature_cols, from_log, EPS, macro_fresh
from .baselines import BASELINES

def _gb():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=250, min_samples_leaf=20, random_state=0)

def wape(y, p): y = np.asarray(y, float); return float(np.abs(y - np.asarray(p, float)).sum() / max(y.sum(), 1e-9))

def _fit_predict(train, test, with_macro):
    cols = feature_cols(train, with_macro)
    cols = [c for c in cols if train[c].nunique(dropna=True) > 1]   # empty or constant columns carry no information and break some library versions
    m = _gb().fit(train[cols], train["resid"])
    return from_log(m.predict(test[cols]) + test["level"].values, test["med"].values), m, cols

def backtest(Y, events, macro=None, horizons=(1, 4), n_origins=13, step=4, min_train_rows=150, sector=None):
    """Returns (summary dict, predictions DataFrame). Each origin retrains on data known at that origin."""
    preds, warnings = [], []
    for h in horizons:
        B = build_rows(Y, h, events, None, sector=sector)
        C = build_rows(Y, h, events, macro, sector=sector) if macro is not None else None
        n = len(Y); last = n - h - 1
        origins = [last - step * k for k in range(n_origins)][::-1]
        if C is not None:   # guard: outside data must actually cover the test weeks, or the test is meaningless
            mc = [c for c in C.columns if c.endswith("_3m")]
            t0 = Y.index[0]
            fresh = np.mean([macro_fresh(macro, t0 + pd.Timedelta(days=7 * (o + 1))) for o in origins])
            cover = float(C[C.origin.isin(origins)][mc].notna().all(axis=1).mean()) * float(fresh) if mc else 0.0
            if cover < 0.5:
                warnings.append(f"The outside data is current for only {cover:.0%} of the weeks used for testing, so it was not tested. "
                                "Check that its dates overlap your sales dates.")
                C = None
        for o in origins:
            tb = B[B["target"] <= o]; te = B[B["origin"] == o]
            if len(tb) < min_train_rows or te.empty: continue
            pb, _, _ = _fit_predict(tb, te, False)
            base = dict(h=h, origin=o)
            for r, p in zip(te.itertuples(), pb):
                preds.append(dict(base, biz=r.biz, model="B", y=r.y_true, p=p, med=r.med))
                preds.append(dict(base, biz=r.biz, model="BL", y=r.y_true, p=(p + r.naive4) / 2, med=r.med))
            for r in te.itertuples():
                preds.append(dict(base, biz=r.biz, model="A1_naive4", y=r.y_true, p=r.naive4, med=r.med))
                if not np.isnan(r.snaive): preds.append(dict(base, biz=r.biz, model="A2_seasonal52", y=r.y_true, p=r.snaive, med=r.med))
                for nm, fn in BASELINES.items():
                    p = fn(Y[r.biz].values[:o + 1], h)
                    if np.isfinite(p): preds.append(dict(base, biz=r.biz, model=nm, y=r.y_true, p=p, med=r.med))
            if C is not None:
                tc = C[C["target"] <= o]; tec = C[C["origin"] == o]
                pc, _, _ = _fit_predict(tc, tec, True)
                for r, p in zip(tec.itertuples(), pc): preds.append(dict(base, biz=r.biz, model="C", y=r.y_true, p=p, med=r.med))
    P = pd.DataFrame(preds)
    S = summarize(P)
    if warnings:
        S["warnings"] = sorted(set(warnings))
        S["verdict"] = {"decision": "NOT_TESTED", "why": S["warnings"][0]}
    return S, P

def _conformal(P, model, h, level=0.8, min_resid=40):
    """Online split-conformal in log space: intervals for origin k use residuals from origins < k."""
    d = P[(P.model == model) & (P.h == h)].copy()
    d["r"] = np.log((d.y / d.med) + EPS) - np.log((d.p / d.med) + EPS)
    cov, wid = [], []
    for o in sorted(d.origin.unique()):
        past = d[d.origin < o]["r"].values
        if len(past) < min_resid: continue
        lo_q, hi_q = np.quantile(past, [(1 - level) / 2, 1 - (1 - level) / 2]); cur = d[d.origin == o]
        lo = from_log(np.log(cur.p / cur.med + EPS) + lo_q, cur.med); hi = from_log(np.log(cur.p / cur.med + EPS) + hi_q, cur.med)
        cov += list(((cur.y >= lo) & (cur.y <= hi)).values); wid += list(((hi - lo) / cur.y.clip(lower=1e-9)).values)
    return (float(np.mean(cov)) if cov else None, float(np.median(wid)) if wid else None)

def paired_ci(P, h, a="B", b="C", n_boot=2000, seed=0):
    """Improvement of b over a in WAPE percentage points (positive = b better), two bootstraps."""
    rng = np.random.default_rng(seed)
    d = P[(P.h == h) & P.model.isin([a, b])].pivot_table(index=["biz", "origin"], columns="model", values=["y", "p"])
    if d.empty or (("p", a) not in d) or (("p", b) not in d): return None
    d = d.dropna(); y = d[("y", a)]; ea = (y - d[("p", a)]).abs(); eb = (y - d[("p", b)]).abs()
    df = pd.DataFrame({"y": y, "ea": ea, "eb": eb}).reset_index()
    def stat(x): return (x.ea.sum() - x.eb.sum()) / max(x.y.sum(), 1e-9) * 100
    shops = df.biz.unique(); origins = df.origin.unique()
    bs = [stat(pd.concat([df[df.biz == s] for s in rng.choice(shops, len(shops))])) for _ in range(300)]
    bo = [stat(pd.concat([df[df.origin == o] for o in rng.choice(origins, len(origins))])) for _ in range(300)]
    ci_s = np.percentile(bs, [2.5, 97.5]); ci_o = np.percentile(bo, [2.5, 97.5])
    lo, hi = min(ci_s[0], ci_o[0]), max(ci_s[1], ci_o[1])        # conservative: widest of the two
    return dict(improvement_pp=float(stat(df)), ci_shops=[float(x) for x in ci_s], ci_origins=[float(x) for x in ci_o],
                ci_conservative=[float(lo), float(hi)], n_pairs=int(len(df)))

def honest_eval(P, h, sel_share=0.6, exclude=("C",)):
    """Pick the best method on the earlier origins, then score it on the later, untouched origins."""
    d = P[(P.h == h) & ~P.model.isin(exclude)]
    origins = sorted(d.origin.unique())
    if len(origins) < 6: return None
    cut = int(len(origins) * sel_share); sel, tst = origins[:cut], origins[cut:]
    def ok(df):
        cnt = df.groupby("model").size(); return [m for m, c in cnt.items() if c == cnt.max()]
    ds, dt = d[d.origin.isin(sel)], d[d.origin.isin(tst)]
    cands = [m for m in ok(ds) if m in ok(dt)]
    if len(cands) < 2: return None
    sw = {m: wape(ds[ds.model == m].y, ds[ds.model == m].p) for m in cands}
    pick = min(sw, key=sw.get); tw = {m: wape(dt[dt.model == m].y, dt[dt.model == m].p) for m in cands}
    base = "A1_naive4" if "A1_naive4" in tw else cands[0]
    return dict(picked=pick, selection_wape=sw[pick], test_wape=tw[pick], test_wape_baseline=tw[base],
                baseline=base, best_possible_test=min(tw.values()), n_test_origins=len(tst),
                selection_wape_all=sw, test_wape_all=tw)

def summarize(P):
    S = {"horizons": {}}
    if P.empty: return S
    for h, g in P.groupby("h"):
        S["horizons"][int(h)] = {m: wape(x.y, x.p) for m, x in g.groupby("model")}
        S["horizons"][int(h)]["coverage80"] = {m: _conformal(P, m, h) for m in g.model.unique()}
        ci = paired_ci(P, h)
        if ci: S["horizons"][int(h)]["C_vs_B"] = ci
        he = honest_eval(P, h)
        if he: S["horizons"][int(h)]["honest"] = he
    S["verdict"] = verdict(S); return S

def select_model(S, h, allow_c=False):
    """Pick the history-based model with the lowest out-of-sample WAPE at horizon h (C only if admitted)."""
    r = S["horizons"].get(h) or S["horizons"].get(min(S["horizons"], key=lambda k: abs(k - h)))
    cands = {m: r[m] for m in ("A1_naive4", "A3_ses", "A4_holt", "B", "BL") + (("C",) if allow_c else ()) if m in r}
    return min(cands, key=cands.get)

def verdict(S):
    """Signal admission test 7 (out-of-sample value). Conservative CI must exclude zero."""
    helps = hurts = tested = 0
    for h, r in S["horizons"].items():
        ci = r.get("C_vs_B")
        if not ci: continue
        tested += 1
        if ci["ci_conservative"][0] > 0: helps += 1
        elif ci["ci_conservative"][1] < 0: hurts += 1
    if tested == 0: return {"decision": "NOT_TESTED", "why": "No outside-signal model was run."}
    if helps and not hurts: return {"decision": "ADMIT", "why": f"Improves out-of-sample error in {helps} of {tested} horizons, never worse."}
    if hurts: return {"decision": "REJECT", "why": f"Worse than history-only in {hurts} of {tested} horizons."}
    return {"decision": "INCONCLUSIVE", "why": "Differences are within noise. Keep history-only until more data is available."}

def forecast_future(Y, events, macro, S, horizons=(1, 2, 4), sector=None, level=0.8):
    """Fit on all known data and forecast from the latest week. Uses C only if the verdict is ADMIT."""
    use_macro = (macro is not None) and S.get("verdict", {}).get("decision") == "ADMIT"
    n = len(Y); out = {}
    for h in horizons:
        D = build_rows(Y, h, events, macro if use_macro else None, sector=sector)
        pr = build_rows(Y, h, events, macro if use_macro else None, targets_known=False, pred_origin=n - 1, sector=sector)
        if pr.empty or len(D) < 100: continue
        pt, _, _ = _fit_predict(D, pr, use_macro)
        pick = select_model(S, h, allow_c=use_macro) if S.get("horizons") else "B"
        nv = pr["naive4"].values
        point = {"A1_naive4": nv, "B": pt, "C": pt, "BL": (pt + nv) / 2}.get(pick)
        if point is None:
            p = np.array([BASELINES[pick](Y[b].values, h) for b in pr.biz])
            point = np.where(np.isfinite(p), p, nv)
        out[h] = pd.DataFrame({"biz": pr.biz.values, "h": h, "point": point, "med": pr.med.values,
                               "model": "C" if (pick == "B" and use_macro) else pick})
    F = pd.concat(out.values()) if out else pd.DataFrame()
    return F, use_macro

def attach_intervals(F, P, model=None, level=0.8):
    rows = []
    for h, g in F.groupby("h"):
        mdl = g["model"].iloc[0]; pm = P[P.model == mdl]
        d = pm[pm.h == h]
        if d.empty and not pm.empty:
            hs = pm.h.unique(); d = pm[pm.h == hs[np.argmin(abs(hs - h))]]
        if len(d) < 40: rows.append(g.assign(lo=np.nan, hi=np.nan)); continue
        r = np.log(d.y / d.med + EPS) - np.log(d.p / d.med + EPS)
        lq, hq = np.quantile(r, [(1 - level) / 2, 1 - (1 - level) / 2])
        lp = np.log(g.point / g.med + EPS)
        rows.append(g.assign(lo=from_log(lp + lq, g.med), hi=from_log(lp + hq, g.med)))
    return pd.concat(rows)
