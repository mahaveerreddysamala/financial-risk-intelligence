# Phase 9: Model Monitoring & Drift Detection

Phase 9 adds lightweight monitoring primitives for detecting distribution changes between a reference population and a current scoring population.

## Population Stability Index

`population_stability_index()` builds quantile bins from the reference population and computes PSI against the current population. The default alert threshold is `0.20` for this portfolio implementation.

PSI is used here as a screening signal rather than as a universal production alert standard. Thresholds should be validated against business impact, data volume, and historical behavior before operational use.

## Prediction-Rate Monitoring

`prediction_rate_shift()` compares observed fraud rates between two periods and flags shifts above a configurable absolute-difference threshold.

This helps distinguish feature-distribution drift from changes in the observed outcome rate.

## Feature-Level Report

`drift_report()` evaluates a selected set of numeric model features and returns:

- drift statistic
- alert threshold
- drift flag
- reference/current sample sizes

The output is designed to feed an eventual monitoring dashboard, scheduled job, or alerting service.

## Production Extension Path

A production implementation would persist reference statistics, calculate monitoring windows from the scoring platform, track missingness and category drift, join delayed fraud outcomes, and emit metrics to an observability system such as Prometheus/Grafana or a managed monitoring service.
