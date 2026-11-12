# MSME Decision Intelligence (Stage 2 toolkit)

Forecasts short-term MSME sales with honest ranges, and tests whether outside signals (WPI, CPI, festival calendar)
actually improve forecasts on data the model has not seen.

## Quick start
Non-technical: double-click `start_windows.bat` (Windows) or run `./start_mac_linux.sh` (Mac/Linux). It sets everything up and opens the app in your browser.

Or by hand:
```
pip install -r requirements.txt
python -m pytest tests -q                     # includes no-leakage, login, data isolation and an app flow test
python run_app.py                             # the dashboard (create an account, then load demo data or upload sales)
python run_backtest.py --sales shops.csv \
    --macro data/cpi_india_monthly.csv data/wpi_headline_india.csv --events india --out results.json
python run_backtest.py --uci Online_Retail.xlsx --events uk --origins 14 --step 2 --out results_uci.json
```

## Accounts and storage
Each owner creates an account (email and password, stored as salted PBKDF2 hashes; five failed logins lock the account for 15 minutes). Sales, branch groups, outside data, analysis runs, forecasts and decisions are saved per account in a SQLite file (`msmedi.db`, path set by `MSME_DB`). Every query is scoped to the signed-in account. Owners can erase their data from the sidebar. Sessions expire after 8 hours.

Dashboard: an overview across all businesses and branches, a branch detail view, uploads (re-uploading a date replaces it, so weekly updates and corrections work), history with past forecasts checked against what actually happened, and a decision log. The analysis always uses the full set of backtests and is cached until new data arrives.

Limits of this version: no email verification or password reset, no HTTPS of its own, and SQLite suits a pilot, not a large user base. Free cloud hosts often wipe local files on restart, so accounts and data would be lost there; use a host with persistent storage, or move to a hosted database.

## Plugging in real data (the only thing left)
**Sales file** (CSV or Excel, daily or weekly, any number of branches). Column names are matched flexibly:
`shop_id`, `date`, `sales` (aliases: store, invoice_date, amount, revenue, turnover ...). A file with no shop column is treated as one shop.
The loader aggregates to Monday weeks, drops partial first/last weeks, drops negative amounts and bad dates (counted in the
report), treats interior missing weeks as zero sales, and drops shops with under 40 weeks of history. It prints a data
readiness report and never changes data silently.

**Outside signals** (one or more CSVs): `series, ref_month, value` and optionally `release_date`. If `release_date` is
missing, CPI is assumed published on the 12th and WPI on the 14th of the following month. A value is only used for a
forecast if it had been released by the forecast origin (no leakage, tested).
Ready now: `data/cpi_india_monthly.csv` (Jan-25 to Aug-26) and `data/wpi_headline_india.csv` (Sep-25 to Sep-26).
Helpers: `parse_cpi_annex_iv()` reads MoSPI's Annexure-IV file directly; `macro_from_wide()` reshapes a wide item-by-month
table (for example DPIIT item-wise WPI) into this format.

## What the engine does
- Models: A1 last-4-week average, A2 same week last year, A3 simple exponential smoothing, A4 damped-trend smoothing,
  B gradient boosting on history and calendar, BL blend of B and A1, C = B plus outside signals.
- Rolling-origin backtest: retrained at every origin on data known at that time.
- Intervals: 80% ranges calibrated online from past out-of-sample errors (split conformal, log scale).
- Guard: outside data that is missing or stale for the tested weeks is reported as NOT_TESTED, not as a result.
- Verdict on outside signals: ADMIT only if C beats B with a confidence interval that excludes zero under both a
  shop-level and a week-level bootstrap (the wider one is used). Otherwise INCONCLUSIVE or REJECT.
- Honest check: the method is chosen on earlier test weeks and scored on later ones.
- Forecasts use the model with the best out-of-sample error per horizon, and C only if ADMIT.

## Docs
`docs/MODEL_CARD.md` (what it does, results, limits) and `docs/JUDGE_QA.md` (likely questions with honest answers). `experiments/` holds the outputs behind the reported numbers and `synthetic_worlds.py`, which reproduces the synthetic-shop results.

## Optional: AI wording assistant
Set `MSME_LLM_API_KEY` and `MSME_LLM_MODEL` for a compatible language-model provider (see the header of `msmedi/explain.py`), then tick the box in the app. Output containing any number that is not in the forecast facts is discarded.

## Honest limits
- Synthetic data has planted effects. It validates the pipeline, not the real-world value of outside signals.
- UCI is one UK wholesaler over about one year (no repeat season), used here to test the method on real transactions.
- Festival dates in `msmedi/features.py` should be verified against an official calendar.
- Rupee impact (`msmedi/impact.py`) is a transparent scenario estimate from stated assumptions, not a measured result.
