# Research Methodology

## Primary Research Question

Does adding timely, relevant external context to a small business's own historical data provide measurable incremental improvement in short-term sales forecasting?

---

## Experimental Design

### A — History

Historical business sales.

### B — History + Seasonality

Historical sales + calendar and seasonal patterns.

### C — History + External Context

Historical sales + seasonality + selected external context variables.

The same model family and evaluation framework should be used when comparing the baseline and treatment.

---

## Model Ladder

1. Seasonal-naive baseline
2. Statistical baseline
3. ML model with lag/calendar/external features
4. Sequence model only if justified by data volume and validation results

Complexity must earn its place through out-of-sample improvement.

---

## Primary Target

Future net sales over a predefined forecasting horizon.

---

## Evaluation

Primary metrics:

- MAE
- WAPE

For prediction intervals:

- Coverage
- Interval width
- Calibration

Evaluation uses:

- Time-based splits
- Rolling evaluation where appropriate
- Business-level reporting
- Untouched final test period

---

## Leakage Control

The model must only use information available at the forecast origin.

External variables must use their historical publication/availability vintage.

---

## Risk Signals

A forecast is not automatically a risk alert.

A risk prediction requires a defined event, horizon and reliable label.

Without reliable labels, the output is reported as an anomaly/context signal rather than a precision/recall-based risk prediction.

---

## What-If Analysis

Scenarios are classified as:

### Mechanical
Arithmetic under explicit assumptions.

### Model-supported
Uses validated historical/model relationships.

### Counterfactual
Requires stronger causal assumptions.

Unsupported scenarios are withheld.

---

## Human + AI Boundary

The AI system provides forecasts, signals, uncertainty and structured evidence.

The owner remains the decision-maker.

An LLM is used as an explanation layer rather than as the financial truth engine.

---

## Stage 2 Requirement

Real Indian business-level data is required before making claims about Indian MSME forecasting performance.

---

## North Star

Find the smallest trustworthy evidence set that produces reliable, decision-relevant improvement.
