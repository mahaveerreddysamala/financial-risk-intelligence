# Model Benchmark

This document defines the reproducible fraud-model comparison protocol.

## Models

- Logistic Regression with class-balanced training
- XGBoost with training-set `scale_pos_weight`

## Validation Strategy

Models are evaluated using the chronological train/validation/test split defined in `src/financial_risk/models/split.py`. No future transaction is allowed into the training set.

## Primary Metrics

Because fraud is an imbalanced classification problem, the primary metrics are PR-AUC, recall, precision, and F1. ROC-AUC is reported as a secondary ranking metric.

## Investigation Capacity

The platform also evaluates operating thresholds and top-K investigator review capacity using precision@K and recall@K. Thresholds should be selected using the operational tradeoff between fraud capture and review volume rather than by assuming `0.50` is universally optimal.

## Results

Measured model results will be added here after the dedicated training/evaluation workflow is executed locally. No benchmark values are hard-coded before measurement.
