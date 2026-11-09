# NeuralGCM stochastic 1.4° Pilot: Stage 0

**Status: preflight only. No model inference, checkpoint download, or GPU job has been performed.**

The first pilot is `2024-10-01 00:00 UTC`, 42 days, 6-hour saved intervals including initial time (169 saved lead times), seed 42 and one stochastic realization. This is deliberately a one-member **engineering pilot**, not yet a probabilistic ensemble skill experiment.

The exact checkpoint is `gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl`, documented by the official NeuralGCM project. The model uses `neuralgcm.PressureLevelModel.from_checkpoint()`, `encode()` with a JAX random key, then `unroll()`, rather than the existing Earth2Studio `run.deterministic()` driver.

## Critical scientific decisions

1. **Boundary forcing:** NeuralGCM requires SST and sea-ice concentration. Stage 0 specifies persistence of initial values, consistent with the official weather-forecast example, but this is only a baseline. A 42-day skill comparison should assess this policy and alternatives that do not leak future observed information. This requirement is not satisfied by simply supplying later ERA5 fields.
2. **ERA5 preparation:** The checkpoint's `input_variables`, `forcing_variables`, grid and coordinate structure must be inspected after loading. Official examples use ARCO ERA5, a 24-hour temporal shift for forcing variables, conservative regridding to the native Gaussian grid, and filling invalid grid cells. These operations must be reproduced and tested for the chosen initialization before inference.
3. **Output contract:** Do not assume an exact match to GraphCast's 70 variables, Pangu6's 69, or SFNO's 73. Explicitly validate available humidity and wind fields, pressure levels, units, longitude/latitude conventions, and the interpretation of lead zero. In particular, do not assume the 1.4° checkpoint has an equivalent `tp06` field.
4. **Ensemble reproducibility:** Record checkpoint hash, Python/JAX/NeuralGCM versions, hardware, RNG seeds, initial-state provenance, forcing policy, model step size and each output variable's units. A single seed does not characterize stochastic uncertainty.
5. **Isolation:** Keep new outputs under `/scratch/$USER/weather-ai/neuralgcm-pilot/`. No changes to existing forecast output trees, converters, or QC products. Do not launch the full 607-initialization operation before a completed and inspected pilot.

## Read-only validation

From the repository root, using ordinary Python:

```bash
python3 scripts/neuralgcm_preflight.py
python3 -m unittest discover -s tests -p 'test_neuralgcm_preflight.py' -v
```

The preflight only validates static settings. It does **not** test NeuralGCM package compatibility, checkpoint downloads, JAX/CUDA functionality, input regridding, runtime stability, or actual NetCDF/Zarr writing. Those are subsequent stages.

## Official references

- [Checkpoint descriptions](https://github.com/neuralgcm/neuralgcm/blob/main/docs/checkpoints.md)
- [Forecasting quick start](https://neuralgcm.readthedocs.io/en/latest/inference_demo.html)
- [NeuralGCM installation](https://neuralgcm.readthedocs.io/en/latest/installation.html)


## Stage 1: isolated GPU smoke runner (unverified on Sapelo2)

The new `scripts/run_neuralgcm_pilot.py` follows the official NeuralGCM ERA5 forecast example: load the 1.4° stochastic checkpoint, open the public ARCO ERA5 Zarr, apply the official **24-hour temporal shift to forcing variables**, select a single initialization, conservatively regrid to the model Gaussian grid, fill missing values, encode with an explicit RNG key, and run an initial **6-hour** GPU forecast. A one-time forcing slice means **persistent initial SST and sea ice** throughout the rollout.

**Important:** This is implementation code, not evidence of successful data access, regridding, or inference. The default command is limited to a six-hour pilot, not 42 days. Outputs are new Zarr directories restricted to `/scratch/$USER/weather-ai/neuralgcm-pilot/`. Existing models and their forecast data are not modified.

From Sapelo2 after the NeuralGCM Pixi environment has been set up:

```bash
pixi run -e neuralgcm python -m unittest discover -s tests -p 'test_neuralgcm_pilot.py' -v
pixi run -e neuralgcm python scripts/run_neuralgcm_pilot.py --forecast-hours 6
```

The second command **does perform** remote ERA5 reads, downloads the checkpoint, uses the GPU, and writes a Zarr output. It may fail if remote Zarr APIs, dependencies, or checkpoint requirements differ from the official example. Inspect diagnostics and the resulting dataset before increasing the horizon.

For 42 days, the forecast lead times are 6, 12, ..., 1008 hours (**168 forecast outputs**); lead zero must be produced separately if a 169-time-point dataset is required. We use `start_with_input=False` to avoid the off-by-one final state advancement inherent in requesting 169 steps with `start_with_input=True`.

**Scientific limitation:** This particular checkpoint is not the precipitation-specific NeuralGCM checkpoint. Do not assume its decoded outputs include precipitation or that its raw outputs match GraphCast/Pangu/SFNO variable inventories. Its wind and moisture fields must be checked before IVT/AR postprocessing. Static SST/sea-ice persistence is a deliberate, potentially impactful 42-day forcing assumption.

Sources: [official forecast quick start](https://neuralgcm.readthedocs.io/en/latest/inference_demo.html), [trained model discussion](https://neuralgcm.readthedocs.io/en/latest/deepdive_into_models.html), and [data preparation](https://neuralgcm.readthedocs.io/en/latest/data_preparation.html).

## Stage 1b: asynchronous Zarr writing (awaiting GPU verification)

NeuralGCM now uses `scripts/neuralgcm_async_io.py`, a NeuralGCM-specific writer rather than the Earth2Studio `AsyncZarrBackend`. The model is unrolled in bounded chunks (default: 8 saved steps = 48 hours), and each chunk is transferred to host and submitted to a **single background I/O worker**. This overlaps GPU inference for the next chunk with serialized Zarr append operations for the previous chunk. `--max-pending-writes 2` bounds outstanding writes and host memory. This is real asynchronous, overlapped I/O but **not 4-way parallel writes**: a single writer avoids conflicting concurrent appends to the same Zarr store.

The writer uses `<destination>.partial.zarr`; it waits for all writes, reopens the store to check expected time length and variables, and renames it to the requested final destination only after success. A failure does not publish a completed output. Partial stores are deliberately retained for diagnosis, and existing output/partial paths block accidental overwrites.

To test without rerunning the earlier completed 6-hour pilot:

```bash
pixi run -e neuralgcm python -m unittest discover -s tests -p 'test_neuralgcm_async_io.py' -v
pixi run -e neuralgcm python -m unittest discover -s tests -p 'test_neuralgcm_pilot.py' -v
pixi run -e neuralgcm python scripts/run_neuralgcm_pilot.py \
  --forecast-hours 12 --write-chunk-steps 1 \
  --output /scratch/$USER/weather-ai/neuralgcm-pilot/neuralgcm_20241001_async_12h.zarr
```

This 12-hour smoke test forces **two separate writes** (6h then 12h), so it exercises the asynchronous append and final publication. It downloads ERA5 and the official checkpoint and runs on GPU. The prior 6-hour Zarr remains untouched. Full 42-day results have **not** been produced or benchmarked.
