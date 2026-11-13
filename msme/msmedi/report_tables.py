"""Markdown tables from a backtest summary, ready to paste into the report."""
NAMES = {"A1_naive4": "4-week average", "A2_seasonal52": "Same week last year", "A3_ses": "Simple smoothing",
         "A4_holt": "Damped-trend smoothing", "B": "Boosted model", "BL": "Blend (boosted + average)", "C": "Boosted + outside data"}

def results_markdown(S, title="Backtest results"):
    out = [f"## {title}", ""]
    hs = sorted(S["horizons"])
    models = [m for m in NAMES if any(m in S["horizons"][h] for h in hs)]
    out += ["| Method | " + " | ".join(f"{h} wk WAPE" for h in hs) + " |", "|---|" + "---|" * len(hs)]
    for m in models:
        row = [f"{S['horizons'][h][m]:.1%}" if m in S["horizons"][h] else "-" for h in hs]
        out.append(f"| {NAMES[m]} | " + " | ".join(row) + " |")
    out += ["", "**Range coverage (nominal 80%)**", "", "| Method | " + " | ".join(f"{h} wk" for h in hs) + " |", "|---|" + "---|" * len(hs)]
    for m in models:
        cells = []
        for h in hs:
            c = S["horizons"][h].get("coverage80", {}).get(m)
            cells.append(f"{c[0]:.0%}" if c and c[0] is not None else "-")
        out.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    he = [(h, S["horizons"][h]["honest"]) for h in hs if "honest" in S["horizons"][h]]
    if he:
        out += ["", "**Honest check: method chosen on earlier weeks, scored on later ones**", "", "| Horizon | Chosen | Error on later weeks | Baseline (4-week average) |", "|---|---|---|---|"]
        for h, x in he: out.append(f"| {h} wk | {NAMES.get(x['picked'], x['picked'])} | {x['test_wape']:.1%} | {x['test_wape_baseline']:.1%} |")
        out += ["", "**Later-week error of every method (not only the one chosen)**", "", "| Method | " + " | ".join(f"{h} wk" for h, _ in he) + " |", "|---|" + "---|" * len(he)]
        for m in NAMES:
            if any(m in x.get("test_wape_all", {}) for _, x in he):
                out.append(f"| {NAMES[m]} | " + " | ".join(f"{x['test_wape_all'][m]:.1%}" if m in x.get("test_wape_all", {}) else "-" for _, x in he) + " |")
    v = S.get("verdict")
    if v: out += ["", f"**Outside-data verdict:** {v['decision']}. {v['why']}"]
    return "\n".join(out) + "\n"
