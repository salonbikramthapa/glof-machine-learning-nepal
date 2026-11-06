# Nepal historical PDGL classification

Machine Learning Approaches for Glacial Lake Outburst Flood Hazard Assessment in the Nepal Himalaya — Salon Bikram Thapa, Coventry University.

This repository contains the source package for the study. It contains cleared-output notebooks, source, environment specifications, aggregate experimental results and source metadata. Raw geospatial/climate data, lake-level derived tables/predictions and fitted models are intentionally absent. The complete private handover supports immediate saved-result inspection.

## Scientific scope

The supervised target is historical ICIMOD PDGL membership (21 listed, 2,045 unlisted in the boundary-screened cohort). Unlisted does not mean safe. LR, KNN, DT and RF are compared with a majority baseline, using nested stratified validation and a separate climate-cell-grouped sensitivity design. This does not forecast future flood dates/probabilities or demonstrate climate causation. The saved protocol and aggregate results document the implemented comparison.

## Set up

Use Python 3.11. From `GLOF_model`, create a virtual environment, install `requirements-lock.txt`, register `python -m ipykernel install --user --name glof --display-name "Python (GLOF)"`, then launch `python -m notebook`. The lock captures the tested macOS environment; a clean cross-platform install remains untested. Do not substitute GUI dependencies into the training environment.

For the app use a separate virtual environment with `requirements-gui.txt`, then `python -m streamlit run app.py --server.address=127.0.0.1`. The app needs lake-level inputs and predictions that this repository excludes. It will not be fully runnable until those are restored from a permitted local handover or regenerated. Aggregate comparison CSVs can be read immediately without fitting models.

## Acquire inputs and preserve layout

Place `outputs/` beside `GLOF_model/` as supplied. Follow source agreements yourself; do not bypass login or licence acceptance.

1. **2015 ICIMOD base inventory:** https://rds.icimod.org/metadata/777e6cb6-c2db-47ab-852c-48bd0f24a158 ; DOI https://doi.org/10.26066/RDS.1971946 . Extract the complete shapefile with all companion files below `outputs/GLOF_MSc_datasets/00_ICIMOD_base_2015/`.
2. **ICIMOD PDGL:** https://doi.org/10.26066/RDS.1971950 . Extract the complete shapefile below `outputs/GLOF_MSc_datasets/01_ICIMOD_PDGL/`. Both ICIMOD sources state CC BY 4.0; preserve provider citation and any download agreement. Retained provider metadata contains additional exact identifiers.
3. **Boundary:** obtain the exact archived NPL ADM0 2019 research boundary identified in `GLOF_model/data/boundary/metadata.json` and `SOURCE.md`. Save it as `nepal_adm0.geojson` in that directory. A newer boundary is a changed research input and may change membership; do not silently substitute it.
4. **ERA5-Land monthly:** https://doi.org/10.24381/cds.68d2bb30 . Accept the CDS licence with your account; request monthly averaged reanalysis, 2m temperature and total precipitation, all months, 00:00. The verified source was global January 1990–August 2026, NetCDF, saved below `outputs/GLOF_MSc_datasets/05_ERA5_Land/`. The model uses only 2005–2014. A smaller request may be scientifically sufficient but is not byte-identical reproduction; the code expects `valid_time`, `latitude`, `longitude`, `t2m`, `tp` and monthly stream `moda` metadata. Use the archived source for exact checks. See source notes for unit/accumulation conventions and CC-BY attribution.
5. **Separate GLO extension:** https://doi.org/10.5281/zenodo.19370146 , version 2; acquire `S2_20172024_NTB_GLOID_v1.02.gpkg` and save under `outputs/GLOF_MSc_datasets/03_GLO_2017_2024/`. CC BY 4.0; preserve version-specific citation. GLO IDs are not ICIMOD IDs.

No raw data are bundled here despite some sources permitting redistribution. Attribution, source terms and exact versions remain necessary. Do not interpret provider notes describing past downloads as evidence those files are inside this source-only package.

## Rebuild or review

For the supplied complete private handover, run notebooks 00–06 with `RUN_TRAINING=False` to inspect saved work. In this source-only repository, first acquire inputs and create the missing outputs:

- Run 00–03 in order to check environment, inspect inventories, apply the boundary and derive dated climate.
- Run 06 before reporting if no recent-extension results exist.
- Back up any existing results. In notebook 04 deliberately set `RUN_TRAINING=True` to regenerate the specified evaluations, because fitted models/predictions are not bundled here. Restore False afterwards.
- Run 05 for reports, then `python src/supervisor_results.py` for added accuracy/error summaries and confusion matrices.
- Check the frozen protocol and verification outcomes; investigate differences instead of removing protective checks. Exact reproduction requires the recorded inputs/software.
- The app then reads the generated files. Its 76 selector/view combinations can be tested using `python src/test_results_app.py` in the separate GUI environment.

Do not tune to held-out scores or present a cross-validation estimator as a deployment model. The complete local handover includes the additional data and saved estimators needed to inspect results without rebuilding.

## Files retained

- `GLOF_model/notebooks/`: seven stages of the study, with saved outputs cleared for sharing.
- `GLOF_model/src/` and `app.py`: analysis, verification and app code.
- `requirements-lock.txt`: exact modelling dependencies; `requirements-gui.txt`: separate app dependencies. These are needed for installation.
- `results/`: aggregate metrics, figures and the fixed experimental protocol.
- `DATA_DICTIONARY.md`, boundary metadata and `outputs/`: data definitions, attribution and acquisition instructions.
- `reports/`: empty directory for checks generated when running the project.


## Licence and data access

Dataset licences and attribution are retained with source metadata. No code licence has been granted in this repository. Raw inputs and saved estimators must be acquired separately before full execution.

## Project links

- Source code and aggregate results: https://github.com/salonbikramthapa/glof-machine-learning-nepal
- Project demonstration: https://youtu.be/dUARnmDIJw0

The demonstration script remains outside this repository.
