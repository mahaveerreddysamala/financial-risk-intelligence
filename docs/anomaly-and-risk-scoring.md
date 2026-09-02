# Anomaly Detection & Ensemble Risk Scoring

## Purpose

This phase adds an unsupervised anomaly signal and combines it with supervised fraud probability and network/velocity risk signals to produce an operational risk score.

## Anomaly Detection

Isolation Forest is trained on training observations only. The model consumes behavioral, velocity, transaction, and network-reuse features. The anomaly score is normalized to `[0, 1]`, with larger values indicating more anomalous behavior.

## Ensemble Risk Score

The default risk composition is:

| Signal | Weight |
|---|---:|
| Supervised fraud probability | 50% |
| Anomaly score | 20% |
| Network risk | 20% |
| Velocity risk | 10% |

All input signals are normalized to `[0, 1]` and weights must sum to 1.

## Operational Decisions

| Risk score | Level | Action |
|---|---|---|
| `< 0.30` | LOW | Approve |
| `0.30–<0.60` | MEDIUM | Monitor |
| `0.60–<0.80` | HIGH | Step-up verification |
| `>= 0.80` | CRITICAL | Hold and investigate |

These thresholds are initial operating rules for the portfolio system. They are not claimed to be production regulatory thresholds and will be evaluated later against business-cost assumptions.
