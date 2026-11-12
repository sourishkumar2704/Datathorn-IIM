"""MSME Decision Intelligence: customer dashboard. Start with:  python run_app.py"""
import json, os, time
import numpy as np, pandas as pd, streamlit as st
from msmedi import auth, store
from msmedi.engine import select_model
from msmedi.explain import build_facts, llm_rewrite, template
from msmedi.features import EVENT_SETS
from msmedi.impact import Z, working_capital_estimate
from msmedi.insights import alerts, realized_accuracy, track_record
from msmedi.io import load_macro_csv, read_sales_rows, to_weekly
from run_backtest import run, to_json

store.init()
st.set_page_config(page_title="MSME Decision Intelligence", layout="wide")
st.markdown("<style>footer{visibility:hidden} #MainMenu{visibility:hidden} .stDeployButton{display:none}</style>", unsafe_allow_html=True)
SESSION_HOURS = 8
S_ = st.session_state


# ---------------------------------------------------------------- login
def gate():
    uid = S_.get("uid")
    if uid and time.time() - S_.get("last", 0) > SESSION_HOURS * 3600:
        S_.clear(); uid = None; st.info("Your session expired. Please log in again.")
    if uid:
        S_["last"] = time.time(); return uid
    st.title("MSME Decision Intelligence")
    st.caption("Sales forecasts for every branch you own, with honest ranges")
    t1, t2 = st.tabs(["Log in", "Create account"])
    with t1:
        e = st.text_input("Email", key="li_e"); p = st.text_input("Password", type="password", key="li_p")
        if st.button("Log in", type="primary"):
            u, err = auth.login(e, p)
            if err: st.error(err)
            else: S_["uid"] = u["id"]; S_["last"] = time.time(); st.rerun()
    with t2:
        n = st.text_input("Your name", key="rg_n"); e2 = st.text_input("Email", key="rg_e")
        p1 = st.text_input("Password (8+ characters, mix letters and numbers)", type="password", key="rg_p1")
        p2 = st.text_input("Repeat password", type="password", key="rg_p2")
        if st.button("Create account", type="primary"):
            if p1 != p2: st.error("Passwords do not match.")
            else:
                uid, err = auth.register(e2, n, p1)
                if err: st.error(err)
                else: S_["uid"] = uid; S_["last"] = time.time(); st.rerun()
    st.stop()


uid = gate()
user = store.get_user(uid)
settings = store.get_settings(uid)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader(f"Hello, {user['name']}")
    ev_opts, lang_opts = ["india", "uk", "none"], ["English", "हिन्दी"]
    events = st.selectbox("Festival calendar", ev_opts, index=ev_opts.index(settings.get("events", "india")))
    lang_label = st.radio("Summary language", lang_opts, index=0 if settings.get("lang", "en") == "en" else 1, horizontal=True)
    lang = "hi" if lang_label == "हिन्दी" else "en"
    if events != settings.get("events", "india") or lang != settings.get("lang", "en"):
        store.set_settings(uid, events=events, lang=lang)
    configured = bool(os.environ.get("MSME_LLM_API_KEY") and os.environ.get("MSME_LLM_MODEL"))
    use_llm = st.checkbox("Polish the wording with an AI assistant", value=False, disabled=not configured,
                          help="Needs MSME_LLM_API_KEY and MSME_LLM_MODEL. The assistant only rewords the facts. It never calculates.")
    with st.expander("Account"):
        o = st.text_input("Current password", type="password", key="pw_o"); n_ = st.text_input("New password", type="password", key="pw_n")
        if st.button("Change password"):
            err = auth.change_password(uid, o, n_)
            st.error(err) if err else st.success("Password changed.")
        st.divider()
        ok = st.text_input("Type DELETE to erase all your saved sales and results", key="del_c")
        if st.button("Erase my data") and ok == "DELETE":
            store.delete_user_data(uid); S_["fp"] = None; st.success("All your data was erased."); st.rerun()
    if st.button("Log out"):
        S_.clear(); st.rerun()


# ---------------------------------------------------------------- data and analysis
def fingerprint():
    if S_.get("fp") is None: S_["fp"] = store.fingerprint(uid, extra=events)
    return S_["fp"]


def restore(summary):
    s = dict(summary); s["horizons"] = {int(k): v for k, v in s["horizons"].items()}; return s


@st.cache_data(show_spinner=False)
def weekly(_uid, fp):
    sales = store.get_sales(_uid)
    return (None, None) if sales.empty else to_weekly(sales, min_weeks=40)


def analysis(Y, rep, force=False):
    fp = fingerprint(); last = store.latest_run(uid)
    if last and last["fingerprint"] == fp and not force:
        r = last["results"]; P = last["preds"]; P["biz"] = P["biz"].astype(str)
        F = pd.DataFrame(r["F"]); F["biz"] = F["biz"].astype(str)
        return restore(r["summary"]), P, F, r["used_outside_signals"], last["created_at"]
    with st.spinner("Testing methods on your past weeks to pick the most accurate one. This can take a few minutes with many branches."):
        S, P, F, used = run(Y, events, store.get_macro(uid), (1, 4), 13, 4)
    res = to_json(S, F, used, rep, Y, events); res["F"] = json.loads(F.to_json(orient="records")) if not F.empty else []
    lw = Y.index[-1]
    rows = [(str(r.biz), int(r.h), str((lw + pd.Timedelta(days=7 * int(r.h))).date()), float(r.point),
             None if r.lo != r.lo else float(r.lo), None if r.hi != r.hi else float(r.hi), r.model) for r in F.itertuples()]
    store.save_run(uid, fp, res, P, rows)
    return S, P, F, used, "just now"


Y, rep = weekly(uid, fingerprint())
ctx = None
if Y is not None and Y.shape[1] > 0:
    S, P, F, used, ran_at = analysis(Y, rep)
    ctx = True

tabs = st.tabs(["Overview", "Branch", "Upload data", "History", "Decisions"])
bz = store.get_branches(uid); bmap = dict(zip(bz.branch, bz.business)) if not bz.empty else {}

# ---------------------------------------------------------------- overview
with tabs[0]:
    st.title("MSME Decision Intelligence")
    if Y is None:
        st.info("Welcome. Add your sales in the **Upload data** tab. You can load demo data there to look around first.")
    elif not ctx:
        st.warning("No branch has 40 full weeks of sales yet, which the tests need. Keep adding weeks in the Upload data tab.")
    else:
        businesses = sorted({bmap.get(c, "My business") for c in Y.columns})
        scope = st.selectbox("Show", ["All businesses"] + businesses)
        cols = [c for c in Y.columns if scope == "All businesses" or bmap.get(c, "My business") == scope]
        rows = []
        for c in cols:
            f = F[F.biz == c].sort_values("h")
            if f.empty: continue
            e = f.iloc[-1]; rec = float(Y[c].tail(4).mean())
            _, w = track_record(P, Y, c, int(e.h) if int(e.h) in set(P.h) else 4, e.model if e.model in set(P.model) else "A1_naive4")
            rows.append({"Branch": c, "Business": bmap.get(c, "My business"), "Recent weekly avg": rec, "Expected": float(e.point),
                         "Change %": (e.point / rec - 1) * 100 if rec > 0 else np.nan, "Low": e.lo, "High": e.hi,
                         "Method": e.model, "Past error %": None if w is None else w * 100})
        T = pd.DataFrame(rows)
        if T.empty:
            st.warning("Not enough history to forecast the selected branches.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            rec_t, exp_t = T["Recent weekly avg"].sum(), T["Expected"].sum()
            k1.metric("Recent weekly sales (4-week avg)", f"{rec_t:,.0f}")
            k2.metric("Expected weekly sales, 4 weeks out", f"{exp_t:,.0f}", f"{(exp_t / rec_t - 1) * 100:+.1f}%" if rec_t > 0 else None)
            k3.metric("Branches", len(T)); k4.metric("Last data week", str(Y.index[-1].date()))
            st.subheader("Branches")
            st.dataframe(T.round(1), hide_index=True, width="stretch")
            st.bar_chart(T.set_index("Branch")[["Recent weekly avg", "Expected"]])
            st.subheader("Sales history")
            top = list(Y[cols].tail(52).sum().sort_values(ascending=False).index[:5])
            pick = st.multiselect("Compare branches (leave empty for the total)", cols, default=top if len(cols) > 1 else [])
            st.line_chart(Y[pick].tail(52) if pick else Y[cols].tail(52).sum(axis=1).rename("Total"))
            heads = [(c, k, t) for c in cols[:40] for k, t in alerts(Y[c], F[F.biz == c], Y.index[-1], events, lang)]
            if heads:
                with st.expander(f"Heads-up across your branches ({len(heads)})", expanded=True):
                    for c, k, t in heads[:30]: st.write(f"**{c}:** {t}")
            d1, d2 = st.columns(2)
            out = F.assign(week=lambda d: [(Y.index[-1] + pd.Timedelta(days=7 * int(h))).date() for h in d.h])[["biz", "h", "week", "point", "lo", "hi", "model"]]
            d1.download_button("Forecasts (CSV)", out.round(0).to_csv(index=False), "forecasts.csv", "text/csv")
            d2.download_button("Full report (JSON)", json.dumps(store.latest_run(uid)["results"], default=str, indent=1), "report.json", "application/json")
            st.caption(f"Analysis last run: {ran_at}. It re-runs automatically when you add new data.")

# ---------------------------------------------------------------- branch detail
with tabs[1]:
    if not ctx: st.info("Add sales first.")
    else:
        shop = st.selectbox("Branch", list(Y.columns), key="branch_pick")
        f = F[F.biz == shop].sort_values("h"); last = Y.index[-1]
        if f.empty: st.warning("Not enough history for this branch.")
        else:
            end = f.iloc[-1]
            st.metric(f"Expected sales, week of {(last + pd.Timedelta(days=7 * int(end.h))).date()}", f"{end.point:,.0f}",
                      help=f"Likely range {end.lo:,.0f} to {end.hi:,.0f}")
            chart = pd.DataFrame({"Past sales": Y[shop].dropna().tail(26)})
            for r in f.itertuples():
                d = last + pd.Timedelta(days=7 * int(r.h)); chart.loc[d, "Forecast"] = r.point; chart.loc[d, "Low"] = r.lo; chart.loc[d, "High"] = r.hi
            st.line_chart(chart)
            facts = build_facts(shop, f, S, rep, last, used, Y[shop])
            st.subheader("In plain words")
            text = llm_rewrite(facts) if (use_llm and lang == "en") else None
            if text: st.write(text); st.caption("Wording by an AI assistant. Every number was checked against the forecast facts.")
            else:
                for tag, sentence in template(facts, lang): st.markdown(f"**{tag}.** {sentence}")
            hs = alerts(Y[shop], f, last, events, lang)
            if hs:
                st.subheader("ध्यान दें" if lang == "hi" else "Heads-up")
                for kind, txt in hs: (st.info if kind in ("festival", "rise") else st.warning)(txt)
            pm = f.iloc[-1]
            tr, sw = track_record(P, Y, shop, int(pm.h) if int(pm.h) in set(P.h) else 4, pm.model if pm.model in set(P.model) else "A1_naive4")
            if sw is not None:
                with st.expander("Track record: past forecasts vs what happened"):
                    st.line_chart(tr); st.write(f"On past tests, this branch's forecasts were off by about {sw:.0%} on average.")
            st.dataframe(f[["h", "point", "lo", "hi", "model"]].rename(columns={"h": "weeks ahead", "point": "expected", "lo": "low (80%)", "hi": "high (80%)", "model": "method"}).round(0), hide_index=True)
            with st.expander("Estimated value of better forecasts (your assumptions)"):
                hz = 4 if 4 in S["horizons"] else max(S["horizons"]); r4 = S["horizons"][hz]; best = select_model(S, hz, allow_c=used)
                st.write(f"At {hz} weeks ahead: simple average error {r4['A1_naive4']:.1%}, chosen method ({best}) error {r4[best]:.1%}.")
                c1, c2 = st.columns(2)
                lead = c1.number_input("Supplier lead time (weeks)", 0.5, 12.0, 1.0, 0.5); svc = c2.selectbox("Service level", list(Z), index=1)
                cg = c1.slider("Cost of goods as % of sales", 30, 95, 70) / 100; hold = c2.slider("Yearly cost of holding stock (%)", 5, 40, 18) / 100
                est = working_capital_estimate(float(Y[shop].dropna().tail(26).mean()), r4["A1_naive4"], r4[best], lead, svc, cg, hold)
                if est["capital_freed"] > 0:
                    st.metric("Stock money that could be freed", f"{est['capital_freed']:,.0f}"); st.metric("Yearly holding cost saved", f"{est['yearly_saving']:,.0f}")
                else: st.info("The chosen method is not better than the simple average here, so no saving is claimed.")
                st.caption("An estimate from the settings above, not a measured result. Defaults are placeholders: change them to match your business.")
        st.subheader("Does outside data help?")
        v = S["verdict"]; st.write(f"**{v['decision']}**: {v['why']}")
        rows_ = {h: {k: x for k, x in r.items() if isinstance(x, float)} for h, r in S["horizons"].items()}
        st.dataframe(pd.DataFrame(rows_).T.rename_axis("weeks ahead").round(3))
        hon = {h: r["honest"] for h, r in S["horizons"].items() if "honest" in r}
        if hon:
            with st.expander("Honest accuracy check (method chosen on earlier weeks, scored on later ones)"):
                for h, x in hon.items():
                    st.write(f"{h} week(s) ahead: chose **{x['picked']}**; on {x['n_test_origins']} later test weeks its error was {x['test_wape']:.1%}, "
                             f"against {x['test_wape_baseline']:.1%} for the {x['baseline']} baseline.")

# ---------------------------------------------------------------- upload
with tabs[2]:
    st.subheader("Add sales")
    st.write("Columns: **shop_id** (your branch name), **date**, **sales**. Daily or weekly rows, any number of branches. "
             "Uploading again for the same branch and date replaces the old value, so you can send corrections or weekly updates.")
    st.download_button("Download a sample file format", "shop_id,date,sales\nMain street,2025-01-01,12500\nMain street,2025-01-02,9800\nStation road,2025-01-01,40200\n", "sales_template.csv", "text/csv")
    f = st.file_uploader("Sales file (CSV or Excel)", type=["csv", "xlsx"], key="up_sales")
    if f is not None:
        try:
            rows_, r_ = read_sales_rows(f)
            st.write(f"Read {r_['rows_kept']:,} day-rows for {r_['branches']} branch(es), {r_['start']} to {r_['end']}."
                     + (f" Skipped {r_['bad_dates']} bad dates." if r_["bad_dates"] else "") + (f" Skipped {r_['bad_amounts']} bad amounts." if r_["bad_amounts"] else "")
                     + (f" Skipped {r_['negative_rows_dropped']} negative amounts." if r_["negative_rows_dropped"] else ""))
            if st.button("Save to my account", type="primary"):
                a, u = store.upsert_sales(uid, rows_); S_["fp"] = None
                st.success(f"Saved: {a:,} new and {u:,} updated rows."); st.rerun()
        except ValueError as e: st.error(str(e))
    st.subheader("Outside data (optional)")
    st.write("Monthly price indices: columns **series, ref_month, value**, optional release_date. Used only if they pass the out-of-sample test.")
    mf = st.file_uploader("Outside data files", type=["csv"], accept_multiple_files=True, key="up_macro")
    if mf and st.button("Save outside data"):
        try: store.upsert_macro(uid, load_macro_csv(mf)); S_["fp"] = None; st.success("Saved."); st.rerun()
        except ValueError as e: st.error(str(e))
    st.subheader("Try it with demo data")
    if st.button("Load demo data into my account"):
        from msmedi.synth import generate
        Yd, md, _ = generate(1, 42, n_biz=6)
        long = Yd.rename(columns={c: f"Demo shop {i + 1}" for i, c in enumerate(Yd.columns)}).rename_axis("date").reset_index().melt(id_vars="date", var_name="biz", value_name="sales")
        store.upsert_sales(uid, long); store.upsert_macro(uid, md)
        store.set_businesses(uid, {f"Demo shop {i + 1}": ("Demo group A" if i < 3 else "Demo group B") for i in range(6)}); S_["fp"] = None
        st.success("Demo data loaded (simulated shops)."); st.rerun()
    st.subheader("Your businesses and branches")
    bz = store.get_branches(uid)
    if bz.empty: st.caption("Branches appear here after you save sales.")
    else:
        st.write("Group your branches by business. Type a business name next to each branch.")
        ed = st.data_editor(bz, disabled=["branch"], hide_index=True, key="bz_ed", width="stretch")
        if st.button("Save business groups"):
            store.set_businesses(uid, dict(zip(ed["branch"], ed["business"]))); st.success("Saved."); st.rerun()

# ---------------------------------------------------------------- history
with tabs[3]:
    if not ctx: st.info("History appears after you add sales.")
    else:
        st.subheader("How past forecasts performed on real outcomes")
        ra = realized_accuracy(store.get_forecast_log(uid), Y)
        if ra.empty:
            st.info("Each analysis saves its forecasts. As new weeks of sales arrive, they are compared here with what actually happened.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Forecasts checked", len(ra)); m2.metric("Average error", f"{(ra.point - ra.actual).abs().sum() / ra.actual.sum():.1%}")
            m3.metric("Outcomes inside the range", f"{ra.inside_range.mean():.0%}")
            st.dataframe(ra[["created_at", "branch", "horizon", "target_week", "point", "actual", "error_pct", "inside_range"]].round(1), hide_index=True, width="stretch")
        st.subheader("Monthly sales by branch")
        st.bar_chart(Y.resample("MS").sum().tail(24))
        st.subheader("Analysis runs")
        st.dataframe(store.list_runs(uid), hide_index=True, width="stretch")
        if st.button("Re-run the analysis now"): S_["fp"] = None; analysis(Y, rep, force=True); st.rerun()

# ---------------------------------------------------------------- decisions
with tabs[4]:
    st.subheader("Record a decision")
    names = list(Y.columns) if ctx else list(bz.branch) if not bz.empty else []
    if not names: st.info("Add sales first.")
    else:
        b = st.selectbox("Branch", names, key="dec_b"); dch = st.selectbox("Decision", ["Plan stock for this forecast", "Wait a week", "Need more data"])
        note = st.text_input("Note (optional)")
        if st.button("Save decision"):
            store.add_decision(uid, b, dch, note, str(Y.index[-1].date()) if ctx else ""); st.success("Saved.")
    dd = store.get_decisions(uid)
    if not dd.empty:
        st.dataframe(dd, hide_index=True, width="stretch")
        st.download_button("Download decisions (CSV)", dd.to_csv(index=False), "decisions.csv", "text/csv")
