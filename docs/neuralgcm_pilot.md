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
