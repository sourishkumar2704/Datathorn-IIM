"""Rule-based heads-ups and a track record. Every statement comes from the shop's own numbers."""
import numpy as np, pandas as pd
from .features import EVENT_SETS

TXT = {
 "en": {"dip": "The 4-week forecast is {pct}% below your recent weekly average.",
        "rise": "The 4-week forecast is {pct}% above your recent weekly average.",
        "wide": "The forecast range is wide, so treat the number as a rough guide.",
        "fest_with": "{name} is about {w} week(s) away. Around last year's date your sales ran {pct}% {dir} your usual.",
        "fest_without": "{name} is about {w} week(s) away. We do not have last year's weeks for this shop, so check your own records before planning stock.",
        "above": "above", "below": "below",
        "names": {"diwali": "Diwali", "holi": "Holi", "dussehra": "Dussehra", "eid_al_fitr": "Eid al-Fitr", "raksha_bandhan": "Raksha Bandhan", "christmas": "Christmas"}},
 "hi": {"dip": "4 हफ्ते का अनुमान आपके हाल के साप्ताहिक औसत से {pct}% कम है।",
        "rise": "4 हफ्ते का अनुमान आपके हाल के साप्ताहिक औसत से {pct}% ज़्यादा है।",
        "wide": "अनुमान का दायरा काफी चौड़ा है, इसलिए इस संख्या को मोटा अंदाज़ा मानें।",
        "fest_with": "{name} लगभग {w} हफ्ते दूर है। पिछले साल इसी समय आपकी बिक्री सामान्य से {pct}% {dir} रही।",
        "fest_without": "{name} लगभग {w} हफ्ते दूर है। इस दुकान के पिछले साल के हफ्ते उपलब्ध नहीं हैं, इसलिए स्टॉक की योजना से पहले अपने रिकॉर्ड देखें।",
        "above": "ऊपर", "below": "नीचे",
        "names": {"diwali": "दिवाली", "holi": "होली", "dussehra": "दशहरा", "eid_al_fitr": "ईद-उल-फ़ितर", "raksha_bandhan": "रक्षा बंधन", "christmas": "क्रिसमस"}},
}

def alerts(y, f, last_week, events="india", lang="en"):
    """y: the shop's weekly sales Series (index = week starts). f: its forecast rows. Returns list of (kind, text)."""
    t = TXT[lang]; out = []; y = y.dropna()
    if f is not None and len(f) and len(y) >= 8:
        e = f.sort_values("h").iloc[-1]; recent = y.tail(8).mean()
        if recent > 0:
            r = e.point / recent - 1
            if r <= -0.15: out.append(("dip", t["dip"].format(pct=round(abs(r) * 100))))
            elif r >= 0.15: out.append(("rise", t["rise"].format(pct=round(r * 100))))
        if e.point > 0 and e.hi == e.hi and e.lo == e.lo and (e.hi - e.lo) / e.point > 1.0: out.append(("wide", t["wide"]))
    ev = EVENT_SETS.get(events, {}); end = last_week + pd.Timedelta(days=6)
    for name, dates in ev.items():
        ds = [pd.Timestamp(d) for d in dates]
        up = [d for d in ds if end < d <= end + pd.Timedelta(days=48)]
        if not up: continue
        nxt = min(up); w = max(int(np.ceil((nxt - end).days / 7)), 1)
        prev = [d for d in ds if d < nxt - pd.Timedelta(days=300)]
        label = t["names"].get(name, name)
        done = False
        if prev:
            p = max(prev); wk = p - pd.Timedelta(days=int(p.weekday())); pos = y.index.get_indexer([wk])[0]
            if pos >= 12 and pos < len(y):
                base = y.iloc[pos - 12:pos - 4].mean()
                if base > 0:
                    u = y.iloc[pos - 1:pos + 1].mean() / base - 1
                    out.append(("festival", t["fest_with"].format(name=label, w=w, pct=round(abs(u) * 100), dir=t["above"] if u >= 0 else t["below"]))); done = True
        if not done: out.append(("festival", t["fest_without"].format(name=label, w=w)))
    return out

def track_record(P, Y, biz, h, model):
    """Past forecasts vs what happened for one shop: DataFrame week, actual, forecast. Plus the shop's WAPE."""
    d = P[(P.biz == biz) & (P.h == h) & (P.model == model)].sort_values("origin")
    if d.empty: return pd.DataFrame(), None
    weeks = [Y.index[0] + pd.Timedelta(days=7 * int(o + h)) for o in d.origin]
    tr = pd.DataFrame({"actual": d.y.values, "forecast": d.p.values}, index=weeks)
    return tr, float(np.abs(tr.actual - tr.forecast).sum() / max(tr.actual.sum(), 1e-9))

def realized_accuracy(log, Y):
    """Compare forecasts saved in the past with what actually happened once those weeks have data.
    log: forecast_log DataFrame; Y: weekly sales (index = week starts, columns = branches)."""
    if log is None or log.empty: return pd.DataFrame()
    d = log.copy(); d["target_week"] = pd.to_datetime(d["target_week"])
    d["actual"] = [Y[b].get(w, np.nan) if b in Y.columns and w in Y.index else np.nan for b, w in zip(d.branch, d.target_week)]
    d = d.dropna(subset=["actual"]); d = d[d.actual > 0]
    if d.empty: return d
    d["error_pct"] = (d.point - d.actual) / d.actual * 100; d["inside_range"] = (d.actual >= d.lo) & (d.actual <= d.hi)
    return d
