"""Transparent scenario estimate: what better forecasts could free up in working capital.
Every input is an explicit assumption the owner can change. Nothing here is a measured result.

Logic (standard safety-stock reasoning):
  forecast error spread  ~ 1.25 x MAE, and MAE = WAPE x average weekly sales
  safety stock (in sales value) = z x error spread x sqrt(lead time in weeks)
  inventory value = safety stock x cost-of-goods ratio
  yearly saving = inventory freed x holding-cost rate
"""
from math import sqrt

Z = {"85%": 1.04, "90%": 1.28, "95%": 1.65}

def working_capital_estimate(avg_weekly_sales, wape_before, wape_after, lead_weeks=1.0, service="90%",
                             cogs_ratio=0.70, holding_rate=0.18):
    """Returns dict with stock needed before/after, capital freed, and yearly carrying-cost saving (same currency as sales)."""
    if min(avg_weekly_sales, lead_weeks) <= 0 or not (0 <= wape_after) or not (0 <= wape_before):
        raise ValueError("Sales and lead time must be positive, and errors must be zero or more.")
    z = Z[service]
    def stock(w): return z * 1.25 * w * avg_weekly_sales * sqrt(lead_weeks) * cogs_ratio
    before, after = stock(wape_before), stock(wape_after)
    freed = max(before - after, 0.0)
    return dict(safety_stock_before=before, safety_stock_after=after, capital_freed=freed,
                yearly_saving=freed * holding_rate, improvement_points=(wape_before - wape_after) * 100,
                assumptions=dict(lead_weeks=lead_weeks, service=service, cogs_ratio=cogs_ratio, holding_rate=holding_rate))
