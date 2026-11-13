"""Plain-English explanations built ONLY from structured facts.
Optional wording assistant: set MSME_LLM_API_KEY and MSME_LLM_MODEL (the request format follows the Anthropic messages API).
The language model (optional) rewrites wording; it never calculates. Any number in its output that is not in the
facts causes a fallback to the deterministic template."""
import json, os, re, urllib.request
import numpy as np

def fmt(x): return f"{x:,.0f}"

def build_facts(shop, f, S, rep, last_week, used_signals, hist):
    h = hist.dropna(); rec, prev = h.tail(8), h.tail(16).head(8)
    chg = (rec.mean() / prev.mean() - 1) * 100 if len(prev) == 8 and prev.mean() > 0 else None
    fc = [dict(weeks_ahead=int(r.h), expected=fmt(r.point), low=fmt(r.lo) if r.lo == r.lo else None,
               high=fmt(r.hi) if r.hi == r.hi else None, method=r.model) for r in f.itertuples()]
    return dict(shop=str(shop), last_data_week=str(last_week.date()), weeks_of_history=int(len(h)),
                recent_8wk_avg=fmt(rec.mean()), earlier_8wk_avg=fmt(prev.mean()) if len(prev) == 8 else None,
                change_pct=f"{abs(chg):.0f}" if chg is not None else None,
                direction=("up" if chg and chg > 0 else "down") if chg is not None else None,
                forecasts=fc, outside_signals_verdict=S["verdict"]["decision"], verdict_reason=S["verdict"]["why"],
                outside_signals_used=bool(used_signals),
                short_history=len(h) < 52)

HI = {"tags": {"Observed": "देखा गया", "Forecast": "अनुमान", "Evidence": "प्रमाण", "Caution": "सावधानी"}, "up": "ऊपर", "down": "नीचे"}

def template_hi(facts):
    o = []
    if facts["change_pct"]:
        o.append(("Observed", f"पिछले 8 हफ्तों में औसत साप्ताहिक बिक्री {facts['recent_8wk_avg']} रही, जो उससे पहले के 8 हफ्तों से {facts['change_pct']}% {HI['up'] if facts['direction'] == 'up' else HI['down']} है।"))
    else:
        o.append(("Observed", f"पिछले 8 हफ्तों में औसत साप्ताहिक बिक्री {facts['recent_8wk_avg']} रही।"))
    fc = facts["forecasts"][-1]
    rng = f", संभावित दायरा {fc['low']} से {fc['high']}" if fc["low"] else ""
    o.append(("Forecast", f"आपके आखिरी डेटा के {fc['weeks_ahead']} हफ्ते बाद के हफ्ते में लगभग {fc['expected']} की बिक्री की उम्मीद है{rng}।"))
    v = facts["outside_signals_verdict"]
    if v == "ADMIT": o.append(("Evidence", "कीमत सूचकांक जैसे बाहरी आँकड़ों ने पिछले हफ्तों में सटीकता बढ़ाई, इसलिए इनका उपयोग किया गया है।"))
    elif v == "NOT_TESTED": o.append(("Evidence", "कोई बाहरी आँकड़े जाँचे नहीं गए, इसलिए यह अनुमान सिर्फ आपकी अपनी बिक्री के इतिहास और त्योहार कैलेंडर पर आधारित है।"))
    else: o.append(("Evidence", "कीमत सूचकांक जैसे बाहरी आँकड़ों से पिछले हफ्तों में सटीकता में साफ़ सुधार नहीं दिखा, इसलिए यह अनुमान आपके अपने इतिहास और त्योहार कैलेंडर पर आधारित है।"))
    if facts["short_history"]: o.append(("Caution", f"आपके पास {facts['weeks_of_history']} हफ्तों का इतिहास है, जो एक साल से कम है, इसलिए सालाना मौसमी पैटर्न अभी नहीं सीखे जा सकते।"))
    return [(HI["tags"][k], s) for k, s in o]

def template(facts, lang="en"):
    if lang == "hi": return template_hi(facts)
    return _template_en(facts)

def _template_en(facts):
    """Returns list of (tag, sentence). Tags: Observed, Forecast, Evidence, Caution."""
    o = []
    if facts["change_pct"]:
        o.append(("Observed", f"Average weekly sales over the last 8 weeks were {facts['recent_8wk_avg']}, "
                              f"{facts['change_pct']}% {facts['direction']} from the 8 weeks before."))
    else:
        o.append(("Observed", f"Average weekly sales over the last 8 weeks were {facts['recent_8wk_avg']}."))
    fc = facts["forecasts"][-1]
    rng = f", most likely between {fc['low']} and {fc['high']}" if fc["low"] else ""
    o.append(("Forecast", f"We expect about {fc['expected']} in the week {fc['weeks_ahead']} weeks after your latest data{rng}."))
    v = facts["outside_signals_verdict"]
    if v == "ADMIT": o.append(("Evidence", "Outside data such as price indices improved accuracy on past weeks, so it is used here."))
    elif v == "NOT_TESTED": o.append(("Evidence", "No outside data was tested, so this forecast uses your own sales history and the festival calendar only."))
    else: o.append(("Evidence", "Outside data such as price indices did not clearly improve accuracy on past weeks, so this forecast uses your own history and the festival calendar only."))
    if facts["short_history"]: o.append(("Caution", f"You have {facts['weeks_of_history']} weeks of history, under a year, so yearly seasonal patterns cannot be learned yet."))
    return o

NUM = re.compile(r"\d[\d,]*\.?\d*")
def numbers_ok(text, facts):
    allowed = {n.replace(",", "") for n in NUM.findall(json.dumps(facts))}
    for n in NUM.findall(text):
        n = n.rstrip(".,").replace(",", "")
        if n not in allowed and not (n.isdigit() and int(n) <= 52): return False
    return True

def llm_rewrite(facts, api_key=None, model=None, timeout=20):
    """Optional wording pass. Returns text or None (caller falls back to the template)."""
    key = api_key or os.environ.get("MSME_LLM_API_KEY"); model = model or os.environ.get("MSME_LLM_MODEL")
    if not key or not model: return None
    body = {"model": model, "max_tokens": 400,
            "system": ("You explain a sales forecast to a small shop owner in plain, friendly English, under 110 words. "
                       "Use ONLY the numbers and statements in FACTS, copied exactly. Never calculate, estimate or add numbers. "
                       "Do not claim why sales changed. Do not give advice beyond what FACTS state. Keep any caution."),
            "messages": [{"role": "user", "content": "FACTS:\n" + json.dumps(facts)}]}
    try:
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in json.load(urllib.request.urlopen(req, timeout=timeout)).get("content", []))
        return text.strip() if text and numbers_ok(text, facts) else None
    except Exception:
        return None
