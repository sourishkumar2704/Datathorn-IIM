"""Command line: python run_backtest.py --sales shops.csv --macro data/cpi_india_monthly.csv data/wpi_headline_india.csv --events india
Writes results JSON (metrics, verdict, forecasts) for the report and the demo app."""
import argparse, json, sys
import numpy as np, pandas as pd
from msmedi.io import load_sales, load_uci, load_macro_csv
from msmedi.features import EVENT_SETS
from msmedi.engine import backtest, forecast_future, attach_intervals
from msmedi.report_tables import results_markdown

def run(Y, events, macro, horizons=(1, 4), n_origins=13, step=4, sector=None, forecast_h=(1, 2, 4)):
    S, P = backtest(Y, EVENT_SETS[events], macro, horizons, n_origins, step, sector=sector)
    F, used = forecast_future(Y, EVENT_SETS[events], macro, S, forecast_h, sector=sector)
    F = attach_intervals(F, P) if not F.empty else F
    return S, P, F, used

def to_json(S, F, used, rep, Y, events):
    last = Y.index[-1]
    fc = {}
    for b, g in F.groupby("biz"):
        fc[str(b)] = [dict(h=int(r.h), model=r.model, week=str((last + pd.Timedelta(days=7 * int(r.h))).date()), point=float(r.point),
                           lo=None if np.isnan(r.lo) else float(r.lo), hi=None if np.isnan(r.hi) else float(r.hi)) for r in g.itertuples()]
    hist = {str(c): [float(v) if not np.isnan(v) else None for v in Y[c].values[-26:]] for c in Y.columns}
    return dict(data_report=rep, summary=S, used_outside_signals=used, events=events, last_data_week=str(last.date()),
                forecasts=fc, history_tail=hist)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales"); ap.add_argument("--uci"); ap.add_argument("--macro", nargs="*", default=[])
    ap.add_argument("--events", default="india", choices=list(EVENT_SETS)); ap.add_argument("--out", default="results.json")
    ap.add_argument("--origins", type=int, default=13); ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 4])
    a = ap.parse_args()
    if a.uci: Y, rep = load_uci(a.uci); sector = None
    elif a.sales: Y, rep, sector = load_sales(a.sales)
    else: sys.exit("Give --sales or --uci")
    macro = load_macro_csv(a.macro) if a.macro else None
    print("Data report:", json.dumps(rep, indent=1))
    S, P, F, used = run(Y, a.events, macro, tuple(a.horizons), a.origins, a.step)
    json.dump(to_json(S, F, used, rep, Y, a.events), open(a.out, "w"), indent=1, default=str)
    open(a.out.replace(".json", "_tables.md"), "w").write(results_markdown(S))
    print(json.dumps(S, indent=1, default=str)[:2500]); print("Saved", a.out)
