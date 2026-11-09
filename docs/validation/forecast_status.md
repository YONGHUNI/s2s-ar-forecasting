# Forecast production and validation status

Last reviewed: 2026-11-09

## Experimental protocol

- Initialization period: 2015-10-01 to 2025-04-30
- Seasonal subset: October through April
- Initialization weekdays: Tuesday and Friday
- Expected initializations: 607
- Forecast horizon: 42 days at 6-hour steps (169 lead times including lead 0)
- Configurations: `config/graphcast_operational.yaml`, `config/pangu_operational.yaml`, and `config/sfno_operational.yaml`

## Current inventory and evidence

| Model | Production status | QC evidence | What has **not** been established |
|---|---|---|---|
| GraphCast Operational | 607 `.nc` files counted in `/scratch/$USER/weather-ai/graphcast-operational` | File-count inventory in interactive Sapelo2 session | Exhaustive per-file validation and forecast skill |
| Pangu6 | User reported completion of 607 initializations; files available in the project shared area | `results/qc/pangu6/pangu6_qc_30_sampled_{summary,detailed}.csv`; reviewer feedback | Exhaustive per-file validation and forecast skill |
| SFNO | 607 `.nc` files counted in `/scratch/$USER/weather-ai/sfno-operational`; pilot `.nc` also present | `qc/sfno/sfno_qc_30_sampled_{summary,detailed}.csv`, pilot moisture diagnostic plots and CSV | Exhaustive per-file validation, moisture correction policy, and forecast skill |

The 607-file counts are counts of paths matching `*.nc` and **do not by themselves prove** that every file is complete, scientifically valid, or contains the expected lead times and variables. The 30-file QC is sampled, not exhaustive.

## SFNO negative-moisture findings

The SFNO pilot produced some negative values in total column water vapor (TCWV) and pressure-level specific humidity. In the submitted diagnostics, many affected cells were concentrated over Antarctica; some pressure levels (such as 850 and 925 hPa) can be below the ground in high terrain. Negative humidity values are not physically valid, and the explanations do not amount to a proof of acceptable IVT behavior.

On **2026-10-19**, **Dr. Zhiqi Yang** responded to the user's pilot QC report (email subject: *Re: Update on Pangu-Weather and SFNO Forecasts*). He considered the Pangu-Weather outputs good and SFNO pilot results reasonable, noted that small negative humidity values can arise from extrapolation, saw no major concern at that stage, and **approved proceeding with full SFNO inference**.

This is authorization to proceed with production, **not** scientific validation of every SFNO output or a decision on how to calculate integrated vapor transport (IVT). Before AR/IVT analysis, explicitly define how to handle negative specific humidity, below-surface levels, and any missing or invalid fields. Quantify the sensitivity of IVT and AR diagnostics to that handling, rather than silently modifying original forecasts.

Relevant reproducibility artifacts:

- `scripts/diagnose_sfno_moisture.py`: pilot-specific diagnostic script; currently contains a hard-coded Sapelo2 input path
- `results/qc/sfno_moisture/negative_moisture_0to7d_6h.csv`
- `results/qc/sfno_moisture/*.png`
- `qc/sfno/sfno_qc_30_sampled_summary.csv` and `qc/sfno/sfno_qc_30_sampled_detailed.csv`
- `results/qc/pangu6/pangu6_qc_30_sampled_summary.csv` and `results/qc/pangu6/pangu6_qc_30_sampled_detailed.csv`

## Reproducibility and limitations

1. Preserve the existing NetCDF results. Do not rerun completed forecasts merely to reorganize code or branch history.
2. Log the exact model/configuration revision, checkpoint, data source, and any post-processing transformations for later scientific analysis.
3. Distinguish **configuration/unit tests** from a GPU inference smoke test, a per-file output audit, and scientific forecast verification. They establish different claims.
4. The 42-day rollout extends well beyond conventional short-range forecasts. Skill, especially at long lead times, needs direct evaluation rather than assumption.
5. Scratch storage may be temporary. Confirm durable copies exist before relying on these data for publication.

## Planned next model

NeuralGCM 1.4° stochastic is planned as a **separate inference adapter and output tree**, not yet implemented or validated in this branch. Existing GraphCast, Pangu6, and SFNO forecasts should remain unchanged.
