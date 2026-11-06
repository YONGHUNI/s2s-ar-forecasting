# 10-panel GraphCast / ERA5 animations (standalone figures)

This one-off plotting workflow lives entirely in `fig/`. It does not change forecast generation, SFNO, Pangu or existing QC.

## Requested figure

One GIF per GraphCast initialization: **607 GIFs** if the configured forecast inventory contains all 607 files. Each GIF has **169 frames** (42 days at 6-hour intervals including lead 0) and **10 panels**, laid out as 2 rows × 5 columns:

| Row | Forecast | P90 | P95 | P99 | ERA5 |
|---|---|---|---|---|---|
| T2m | GraphCast | ≥ P90 | ≥ P95 | ≥ P99 | ERA5 reference |
| PR (6h) | GraphCast | ≥ P90 and positive | ≥ P95 and positive | ≥ P99 and positive | ERA5 reference |

Each P90/P95/P99 is computed **once per initialization and variable**, from ALL finite values over the original forecast NetCDF's lead-time, latitude, and longitude dimensions (before spatial sampling for plotting). Thresholds stay fixed across the 169 frames. These are whole-domain spatial-temporal thresholds, **not** climatological or local extreme thresholds. Zero-precipitation cells remain in the precipitation distribution.

T2m is displayed in °C (source K); precipitation is displayed in mm per 6h (source m). Both are thresholded in their original stored units, with the same results as thresholding after a monotone unit conversion. Slight negative precipitation may arise numerically; values below the configured negative tolerance raise an error, otherwise they are clipped to zero for plotting and percentile calculations (the original NetCDF is never changed). Raw negative counts are logged.

ERA5 is the ARCO ERA5 reanalysis data source used as GraphCast's inference initial-condition source, but plotted at the **forecast valid time**, not by repeating the initialization field. Request `t2m` and `tp06` (the latter is the six-hour accumulated value). ERA5 is reanalysis, not an independent ground-station observation.

## Paths and environments

The default forecast root is `/scratch/$USER/weather-ai/graphcast-operational` (the operational, not deterministic subdirectory). Edit `fig/config.yaml` if the forecast files are stored elsewhere. Path tokens `{user}` resolve to the Linux account. Derived data, logs and GIFs go under `/scratch/$USER/weather-ai/fig/`, **not** into the Git repository.

The repository's existing Pixi `graphcast` environment is used to retrieve reference ERA5 via Earth2Studio ARCO; `postprocess` is used to render GIFs. No changes to `pixi.toml`, the lockfile, model inference or existing Slurm files are required.

## Workflow

1. Make sure all 607 GraphCast NetCDF files exist in the configured `forecast.root`. The script validates the inventory.
2. Preview the required ERA5 months:
   `pixi run -e graphcast python fig/prepare_era5.py --config fig/config.yaml --list-months`
3. Prepare the reference monthly NetCDFs, with one Slurm task per month, **before** plotting. This prevents 607 plotting jobs from independently requesting the same remote ERA5 data. Use `bash fig/submit.sh all`.
4. The plotting array starts only if all reference months finish successfully. Each plot task selects a single forecast by deterministic sorted index.
5. To validate one initialization first:
   `pixi run -e postprocess python -m fig.plot_10panel --config fig/config.yaml --index 0`
   (Requires the relevant ERA5 monthly reference files.)

`bash fig/submit.sh prepare` launches only ERA5 preparation; `bash fig/submit.sh plots` launches only the GIF array. Default array concurrency is four preparation jobs and eight plotting jobs, configurable using `PREP_CONCURRENCY` and `PLOT_CONCURRENCY` environment variables. This is bounded CPU parallelism, not 607 concurrent jobs.

## Scientific and operational caveats

* Lead 0's `tp06` refers to the six hours ending at initialization; it is not a zero-hour total.
* Reference cells must match the forecast latitude/longitude coordinates after striding. The plotting script aborts on a mismatch instead of silently plotting misaligned fields.
* A forecast whose metadata or lead schedule differs from 0..1008 hours fails validation. The pipeline never silently drops missing frames.
* Percentile thresholds are initialization-specific. They do not provide common climatological benchmarks across initializations.
* GIF generation and remote downloads can create significant storage and network load. Validate one monthly reference and one GIF before submitting the full production arrays.
* Output files are written through temporary paths and renamed only after successful completion. Existing completed files are skipped, enabling restart.
* Nothing has been run on Sapelo2 by committing this code. Actual server-side results and costs require an execution test.

## Display settings (2026-11 widescreen)
- Fixed **16:9** 2×5 arrangement, variable per row (T2m above precipitation).
- Static Natural Earth 1:110m shoreline outlines appear on the P90/P95/P99 panels to retain geographic context where the threshold mask is sparse. Bundled data are public domain: https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_land.geojson
- The same temperature colour limits, **−40 to 40 °C**, and 6-hour precipitation limits, **0 to 60 mm**, are used in ALL 607 GIFs. Colourbar ends use extends; values outside the displayed range are coloured using saturated end colours, not edited or removed.
- Nonphysical negative forecast precipitation is clipped **for display and quantile calculation only**; original NetCDF untouched. QC logs record the minimum and clipped count. Zero-rain cells count toward thresholds but are omitted from extreme-rainfall overlays.
- Thresholds still vary by initialization because they are calculated from each forecast NetCDF's entire time×space field; only the display colour limits are universal.
- Sapelo-specific rootless Nix bootstrap is included in both figure Slurm scripts; submit helper no longer calls Nix on the submit node.

### Geographic aspect ratio

Each panel uses `imshow(..., aspect="equal")` in the original longitude/latitude coordinate system, instead of stretching pixels with `aspect="auto"`. This preserves the Plate Carrée longitude/latitude ratio (a full-global map is ~2:1 in degrees). The outer GIF canvas remains 16:9, and individual panels may have extra vertical whitespace. Equal degrees do not imply equal ground distances at different latitudes.
