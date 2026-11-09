#!/usr/bin/env python3
"""Count the forecast months required by the ERA5 cache, no Nix required."""
from datetime import datetime, timedelta
from pathlib import Path
import os
import re

raw = Path("fig/config.yaml").read_text(encoding="utf-8")

def field(name):
    found = re.search(r'^\s*' + re.escape(name) + r':\s*"?([^"\n]+)', raw, re.M)
    if not found:
        raise ValueError(f"Missing {name} in fig/config.yaml")
    return found.group(1).strip()

root = Path(field("root").replace("{user}", os.environ["USER"]))
files = sorted(root.glob(field("pattern")))
expected = int(field("expected_initializations"))
if len(files) != expected:
    raise SystemExit(f"Expected {expected} files, got {len(files)}")

months = set()
for path in files:
    match = re.fullmatch(r"graphcast_operational_(\d{8})_42day_subset\.nc", path.name)
    if not match:
        raise ValueError(path.name)
    start = datetime.strptime(match.group(1), "%Y%m%d")
    for lead in range(int(field("steps"))):
        valid = start + timedelta(hours=int(field("hours_per_step")) * lead)
        months.add(valid.strftime("%Y%m"))
print(len(months))
