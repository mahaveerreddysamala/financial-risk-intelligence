# Phase 44 — Advanced Model Monitoring

The platform now produces a unified monitoring snapshot across three control surfaces:

1. **Feature drift** — existing PSI-based feature drift monitoring is reused for the current production window versus a reference population.
2. **Performance degradation** — current PR-AUC, ROC-AUC, precision, recall, and F1 can be compared with a reference baseline. A configurable relative-drop tolerance plus a minimum PR-AUC floor identifies degradation.
3. **Calibration degradation** — current Brier score is compared with the reference score and flagged when calibration quality worsens beyond the configured relative tolerance.

## Monitoring status

`build_monitoring_status()` converts those signals into an alert-ready status:

- `healthy / info`: no detected issue
- `drift / warning`: feature drift without model-quality degradation
- `degraded / warning`: performance or calibration degradation
- `degraded / critical`: multiple quality issues are present together

The `MonitoringStatus` object retains explicit boolean flags and machine-readable reasons for downstream alerting, dashboards, or incident workflows.

## Batch snapshot

`monitor_model_window()` is the orchestration entry point. It returns the feature-drift table, optional performance report, optional calibration report, and aggregate status in one dictionary so a scheduler or monitoring service can persist the complete decision snapshot.

These thresholds are platform monitoring defaults rather than financial-regulatory limits; production values should be calibrated to validated historical behavior and business risk appetite.
