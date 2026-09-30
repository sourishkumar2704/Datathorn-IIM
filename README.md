# UdyamAI — MSME Decision Intelligence

## BAICONF Datathon 2026 — Stage 1

UdyamAI is a proposed AI-based decision-intelligence framework for Indian micro and small businesses.

The research investigates whether adding timely, relevant external context to a business's own historical data can provide measurable incremental improvement in short-term sales forecasting.

## Research Question

> Does adding timely, relevant external context to a small business's own historical data provide measurable incremental improvement in short-term sales forecasting?

## Core Workflow

Forecast → Detect → Simulate → Explain → Owner decides

The system is designed around five layers:

1. Business data validation
2. Short-term forecasting
3. Risk and opportunity signals
4. Evidence-graded what-if scenarios
5. Human decision support

## Research Experiment

### A — History
Business historical sales data.

### B — History + Seasonality
Historical sales + calendar and seasonal patterns.

### C — History + External Context
Historical sales + seasonality + selected external economic/context variables.

The central question is whether C produces reproducible out-of-sample improvement over B.

## Hypotheses

### H0

External context provides no meaningful incremental out-of-sample improvement after accounting for business history and seasonality.

### H1

External context provides measurable and reproducible incremental improvement for at least some validated business segments or forecasting horizons.

## Important Scope

This is a Stage 1 research proposal.

No model results are claimed in this repository.

Indian MSME performance will not be generalized until real Indian business-level data is used for validation.

## Repository

- `sources/` — source registry and direct links
- `methodology/` — research and evaluation framework
- `data/` — proposed data architecture
- `presentation/` — Stage 1 presentation

## Evidence Principle

Not every public dataset becomes a model feature.

Each candidate source is evaluated for:

- credibility
- relevance
- availability
- sector/geographic fit
- temporal coverage
- leakage risk
- data quality
- out-of-sample value

## North Star

> Find the smallest trustworthy evidence set that produces reliable, decision-relevant improvement.