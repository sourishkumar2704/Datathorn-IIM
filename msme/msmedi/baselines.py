"""Standard short-series benchmarks (pure numpy): simple exponential smoothing and damped-trend Holt.
Parameters are chosen per series by one-step-ahead error on the last 52 known weeks, using only data up to the origin."""
import numpy as np

def _recent(y):
    y = np.asarray(y, float); bad = np.where(~np.isfinite(y))[0]
    return y[(bad[-1] + 1 if len(bad) else 0):][-52:]

def ses(y, h=1, alphas=(0.1, 0.2, 0.3, 0.5, 0.7)):
    y = _recent(y)
    if len(y) < 4: return np.nan
    best = None
    for a in alphas:
        l, sse = y[0], 0.0
        for v in y[1:]:
            sse += (v - l) ** 2; l = a * v + (1 - a) * l
        if best is None or sse < best[0]: best = (sse, l)
    return float(max(best[1], 0.0))

def holt_damped(y, h=1, alphas=(0.2, 0.4), betas=(0.05, 0.15), phi=0.9):
    y = _recent(y)
    if len(y) < 8: return np.nan
    best = None
    for a in alphas:
        for b in betas:
            l, t, sse = y[0], (y[3] - y[0]) / 3, 0.0
            for v in y[1:]:
                f = l + phi * t; sse += (v - f) ** 2
                ln = a * v + (1 - a) * f; t = b * (ln - l) + (1 - b) * phi * t; l = ln
            if best is None or sse < best[0]: best = (sse, l, t)
    _, l, t = best
    return float(max(l + sum(phi ** k for k in range(1, h + 1)) * t, 0.0))

BASELINES = {"A3_ses": ses, "A4_holt": holt_damped}
