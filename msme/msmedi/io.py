"""Loading and validating real-world data. Messy shop data is expected: this module
aggregates to weeks, reports problems, and never silently drops data."""
import numpy as np, pandas as pd
from .features import add_release_dates

ALIASES = {
    "biz": ["shop_id", "shop", "store", "store_id", "business", "business_id", "biz", "customer_id", "stockcode"],
    "date": ["date", "invoice_date", "invoicedate", "txn_date", "bill_date", "day", "order_date"],
    "sales": ["sales", "amount", "revenue", "turnover", "total", "value", "sale_amount"],
}

def _rename(df):
    low = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=low); out = {}
    for std, names in ALIASES.items():
        hit = next((n for n in names if n in df.columns), None)
        if hit is None and std == "biz":
            df = df.assign(biz="My shop"); out["biz"] = "biz"; continue
        if hit is None: raise ValueError(f"Missing a '{std}' column. Accepted names: {names}. Found: {list(df.columns)}")
        out[hit] = std
    keep = df.rename(columns=out)
    extra = [c for c in ("sector", "city") if c in keep.columns]
    return keep[["biz", "date", "sales"] + extra]

def to_weekly(df, min_weeks=40, drop_partial_edges=True):
    """df: biz, date, sales (any granularity). Returns (Y wide weekly, report dict)."""
    df = df.copy(); df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rep = {"rows_in": int(len(df)), "bad_dates": int(df["date"].isna().sum())}
    df = df.dropna(subset=["date"]); df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    rep["bad_amounts"] = int(df["sales"].isna().sum()); df = df.dropna(subset=["sales"])
    rep["negative_rows_dropped"] = int((df["sales"] < 0).sum()); df = df[df["sales"] >= 0]
    dmin, dmax = df["date"].min().normalize(), df["date"].max().normalize()
    df["week"] = (df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")).dt.normalize()
    W = df.groupby(["week", "biz"])["sales"].sum().unstack("biz")
    full = pd.date_range(W.index.min(), W.index.max(), freq="7D"); W = W.reindex(full)
    if drop_partial_edges:   # a week only counts if the data cover all of it
        if dmin > W.index[0]: W = W.iloc[1:]
        if dmax < W.index[-1] + pd.Timedelta(days=6): W = W.iloc[:-1]
        rep["partial_edge_weeks_dropped"] = True
    first = W.notna().idxmax(); last = W.notna()[::-1].idxmax()
    for c in W.columns: W.loc[(W.index < first[c]) | (W.index > last[c]), c] = np.nan   # trim, don't invent
    inner_missing = {c: int(W.loc[first[c]:last[c], c].isna().sum()) for c in W.columns}
    # interior gaps: a missing week for a shop that is otherwise reporting is treated as zero sales
    for c in W.columns: W.loc[first[c]:last[c], c] = W.loc[first[c]:last[c], c].fillna(0.0)
    weeks_ok = W.notna().sum()
    short = weeks_ok[weeks_ok < min_weeks].index.tolist()
    W = W.drop(columns=short)
    rep.update(shops_kept=int(W.shape[1]), shops_dropped_short_history=[str(s) for s in short],
               weeks=int(W.shape[0]), start=str(W.index.min().date()), end=str(W.index.max().date()),
               interior_missing_weeks_filled_as_zero=int(sum(v for k, v in inner_missing.items() if k in W.columns)),
               zero_weeks_pct=round(float((W == 0).sum().sum() / max(W.notna().sum().sum(), 1) * 100), 2))
    lw = np.log1p(W); mad = (lw - lw.median()).abs().median()
    rep["outlier_weeks_flagged"] = int((((lw - lw.median()).abs() > 6 * 1.4826 * mad.replace(0, np.nan)).sum()).sum())
    return W, rep

def _read_any(src):
    name = str(getattr(src, "name", src)).lower()
    return pd.read_excel(src) if name.endswith((".xlsx", ".xls")) else pd.read_csv(src)

def read_sales_rows(src):
    """Clean rows ready for storage: (biz, date, sales) summed per day, plus a report of what was skipped."""
    d = _rename(_read_any(src))[["biz", "date", "sales"]].copy()
    rep = {"rows_in": int(len(d))}
    d["date"] = pd.to_datetime(d["date"], errors="coerce", dayfirst=False); rep["bad_dates"] = int(d["date"].isna().sum())
    d["sales"] = pd.to_numeric(d["sales"], errors="coerce"); rep["bad_amounts"] = int(d["sales"].isna().sum())
    d = d.dropna(subset=["date", "sales"]); rep["negative_rows_dropped"] = int((d["sales"] < 0).sum()); d = d[d["sales"] >= 0]
    d["biz"] = d["biz"].astype(str).str.strip().replace("", "(blank)"); d["date"] = d["date"].dt.normalize()
    d = d.groupby(["biz", "date"], as_index=False)["sales"].sum()
    rep.update(rows_kept=int(len(d)), branches=int(d["biz"].nunique()),
               start=str(d["date"].min().date()) if len(d) else None, end=str(d["date"].max().date()) if len(d) else None)
    if d.empty: raise ValueError("No usable rows. Check that dates look like 2025-03-14 and sales are numbers.")
    return d, rep

def load_sales(path, **kw):
    df = _read_any(path)
    d = _rename(df); Y, rep = to_weekly(d[["biz", "date", "sales"]], **kw)
    sector = None
    if "sector" in d.columns:
        sector = d.drop_duplicates("biz").set_index("biz")["sector"].to_dict()
    return Y, rep, sector

def load_uci(path, top_n=30, cache="/tmp/uci_clean.pkl"):
    """UCI Online Retail: product lines (top_n StockCodes by revenue) plus TOTAL as series."""
    import os
    if cache and os.path.exists(cache): d = pd.read_pickle(cache)
    else:
        d = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
        d = d[~d["InvoiceNo"].str.startswith("C") & (d["Quantity"] > 0) & (d["UnitPrice"] > 0)].copy()
        d["sales"] = d["Quantity"] * d["UnitPrice"]
        d = d[~d["StockCode"].str.upper().isin(["POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "AMAZONFEE", "S"])]
        if cache: d.to_pickle(cache)
    top = d.groupby("StockCode")["sales"].sum().nlargest(top_n).index
    s = d[d["StockCode"].isin(top)].rename(columns={"StockCode": "biz", "InvoiceDate": "date"})[["biz", "date", "sales"]]
    t = d.rename(columns={"InvoiceDate": "date"})[["date", "sales"]].assign(biz="TOTAL")
    return to_weekly(pd.concat([s, t]), min_weeks=40)

def load_macro_csv(paths):
    paths = [paths] if isinstance(paths, (str, bytes)) or hasattr(paths, "read") else list(paths)
    m = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True); need = {"series", "ref_month", "value"}
    if not need <= set(m.columns): raise ValueError(f"Macro CSV needs columns {sorted(need)} (optional: release_date). Found {list(m.columns)}")
    return add_release_dates(m)

def parse_cpi_annex_iv(path):
    """MoSPI CPI press-release Annexure-IV (all-India monthly combined index) -> macro frame."""
    x = pd.read_excel(path, header=None); rows = []
    for _, r in x.iterrows():
        lab = r[0]
        try: m = pd.to_datetime(lab) if not isinstance(lab, str) else pd.to_datetime(lab.replace("*", "").strip(), format="%b-%y")
        except Exception: continue
        v = pd.to_numeric(r[3], errors="coerce")
        if pd.notna(v): rows.append(dict(series="cpi", ref_month=m, value=float(v)))
    return add_release_dates(pd.DataFrame(rows))

def macro_from_wide(df, name_col, series_prefix="wpi_", month_cols=None):
    """Melt a wide item-by-month table (e.g. DPIIT item-wise WPI) into the macro schema.
    Column headers must be parseable as months."""
    cols = month_cols or [c for c in df.columns if c != name_col and pd.notna(pd.to_datetime(str(c), errors="coerce"))]
    l = df.melt(id_vars=[name_col], value_vars=cols, var_name="ref_month", value_name="value")
    l["series"] = series_prefix + l[name_col].astype(str).str.lower().str.replace(r"\W+", "_", regex=True).str.strip("_")
    l["ref_month"] = pd.to_datetime(l["ref_month"].astype(str)); l["value"] = pd.to_numeric(l["value"], errors="coerce")
    return add_release_dates(l[["series", "ref_month", "value"]])
