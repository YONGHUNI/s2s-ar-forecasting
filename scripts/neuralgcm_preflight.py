#!/usr/bin/env python3
"""Read-only preflight for a proposed NeuralGCM pilot. Does not run inference."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXPECTED_CHECKPOINT = "gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl"
EXPECTED_FORCING = "persist_initial_sst_and_sea_ice"


def validate_config(data: dict) -> dict:
    if data.get("model") != "NeuralGCM":
        raise ValueError("pilot model must be NeuralGCM")
    if data.get("checkpoint_uri") != EXPECTED_CHECKPOINT:
        raise ValueError("unverified NeuralGCM checkpoint URI")
    if data.get("forcing_policy") != EXPECTED_FORCING:
        raise ValueError("forcing policy must be explicit and reviewed")
    if data.get("status") != "preflight_only":
        raise ValueError("this is a preflight-only configuration")
    initialization = datetime.fromisoformat(data["initialization"].replace("Z", "+00:00"))
    if initialization.utcoffset() != timedelta(0):
        raise ValueError("initialization must be UTC")
    days = data["forecast_days"]
    step = data["output_interval_hours"]
    if not isinstance(days, int) or days < 1:
        raise ValueError("forecast_days must be a positive integer")
    if not isinstance(step, int) or step < 1 or (days * 24) % step:
        raise ValueError("output_interval_hours must divide the forecast horizon")
    if data.get("include_initial_time") is not True:
        raise ValueError("pilot must include lead zero")
    seeds = data.get("ensemble_seeds")
    if not isinstance(seeds, list) or not seeds or any(
        type(seed) is not int or seed < 0 for seed in seeds
    ) or len(seeds) != len(set(seeds)):
        raise ValueError("ensemble_seeds must be nonempty unique nonnegative integers")
    if not isinstance(data.get("output_root_template"), str):
        raise ValueError("output root template required")
    root = data["output_root_template"]
    if root != "/scratch/{user}/weather-ai/neuralgcm-pilot":
        raise ValueError("unexpected output root: do not write into existing model directories")
    count = days * 24 // step + 1
    return {
        "model": data["model"],
        "checkpoint_uri": data["checkpoint_uri"],
        "initialization_utc": initialization.isoformat(),
        "forecast_end_utc": (initialization + timedelta(days=days)).isoformat(),
        "forecast_days": days,
        "output_interval_hours": step,
        "lead_time_count": count,
        "ensemble_members": len(seeds),
        "ensemble_seeds": seeds,
        "forcing_policy": data["forcing_policy"],
        "output_root": root.format(user=os.environ.get("USER", "<user>")),
        "inference_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/neuralgcm_pilot.json"),
    )
    args = parser.parse_args()
    data = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(validate_config(data), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
