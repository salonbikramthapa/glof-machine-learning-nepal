# GLO annual lake inventories, 2017–2024

Downloaded 2026-09-28 from https://zenodo.org/records/19370146 (record version v2, published 2026-04-01; data filenames v1.02). This is the latest version linked by the original record 17802334 at the time of access.

Version-specific citation: Rawlins, L., Watson, C. S., Bhambri, R., Khadka, N., & Chand, M. B. (2026). Datasets supporting - Glacial Lake Observatory (GLO): A dataset of glacial lakes in Nepal and transboundary catchments (2017–2024) (v2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19370146

The unchanged author README also supplies the 2025 concept citation: https://doi.org/10.5281/zenodo.17802333. Retain the version-specific DOI to identify the actual files used. Licence: CC BY 4.0. Full provider metadata and file URLs/checksums are in zenodo_record.json.

All five GeoPackages and the provider README were downloaded. The two deep-learning model ZIP archives and classification script were intentionally omitted; they are training/inference software, not needed to use the requested lake inventory.

## Validation
All six source files match Zenodo's exact byte sizes and MD5 checksums. All five GeoPackages pass SQLite quick_check. Both annual layers contain every year 2017–2024. Checks establish file integrity and readable tables, not a full geometry or scientific quality audit. VALIDATION.json records counts, fields and years.

Sentinel-2: 22,294 annual features, 4,150 unique lake IDs, 4,150 dissolved polygons and 4,150 centroids. Sentinel-1: 18,129 annual features, 2,966 distinct annual lake IDs, but 2,967 dissolved features/IDs. This source-level count discrepancy has not been resolved; retain the raw data and check ID coverage when joining.

## Use
Open S2_20172024_NTB_GLOID_v1.02.gpkg in QGIS for annual lake outlines. AREA_YEAR is the observation year; AREA is km² according to the README. Unique-lake layers contain EXPANSION_RATE and uncertainty/significance fields; they are merged extents rather than an additional observation year.

Stored polygon CRS: ESRI:102025; centroid CRS: EPSG:4326. Filenames are preserved exactly as distributed. Some filenames in the provider README differ slightly from actual archive names (e.g. centroid/centroids and a missing underscore in the S1 unique file). Use the names in DOWNLOAD_MANIFEST.md.

These are the full supplied Nepal/transboundary files, not subsets matched to the user's 3,624-lake ICIMOD inventory. GLO_ID and ICIMOD GL_ID are different ID systems; spatial matching and review are needed. Missing years or missing matches must not automatically be interpreted as zero lake area.
