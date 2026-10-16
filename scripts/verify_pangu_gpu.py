#!/usr/bin/env python3
"""Profile actual ONNX node placement during one Pangu6 forecast (GPU node only)."""
import collections
import json
import os
from datetime import datetime
from pathlib import Path

import onnxruntime as ort

# Intercept ORT session construction before importing Earth2Studio's Pangu.
_original_session = ort.InferenceSession
_sessions = []

def profiled_session(*args, **kwargs):
    options = kwargs.get("sess_options")
    if options is None:
        options = ort.SessionOptions()
        kwargs["sess_options"] = options
    options.enable_profiling = True
    session = _original_session(*args, **kwargs)
    _sessions.append(session)
    print("ORT session providers:", session.get_providers(), flush=True)
    return session

ort.InferenceSession = profiled_session

import numpy as np
from earth2studio import run
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Pangu6

root = Path(f"/scratch/{os.environ['USER']}/weather-ai/pangu-operational/gpu-validation")
root.mkdir(parents=True, exist_ok=True)
job = os.environ.get("SLURM_JOB_ID", "manual")
out = root / f"pangu6_{job}.zarr"
assert not out.exists(), f"Refusing to overwrite {out}"

print("Loading Pangu6 models...", flush=True)
model = Pangu6.load_model(Pangu6.load_default_package())
print("Running one 6-hour step...", flush=True)
run.deterministic([datetime(2024, 10, 1)], 1, model, ARCO(),
                  ZarrBackend(file_name=str(out)), device="cuda")
print("Output:", out, flush=True)

if not _sessions:
    raise RuntimeError("No ONNX Runtime sessions captured; cannot verify device placement")

totals = collections.Counter()
for n, session in enumerate(_sessions):
    profile_file = Path(session.end_profiling())
    events = json.loads(profile_file.read_text())
    counts = collections.Counter(
        e.get("args", {}).get("provider", "unknown")
        for e in events
        if e.get("cat") == "Node" and "provider" in e.get("args", {})
    )
    totals.update(counts)
    dest = root / f"onnx-profile-{job}-{n}.json"
    profile_file.replace(dest)
    print(f"Session {n}: providers={session.get_providers()}, nodes={dict(counts)}, profile={dest}",
          flush=True)

print("ONNX node placement totals:", dict(totals), flush=True)
if totals["CUDAExecutionProvider"] == 0:
    raise RuntimeError("No ONNX nodes documented on CUDA; GPU execution not verified")
print("CUDA ONNX node execution verified.", flush=True)
