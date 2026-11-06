# ERA5-Land monthly climate source

Source accessed 28 September 2026. The NetCDF is not included in this repository.

File: ERA5_Land_monthly_global_t2m_tp_1990_2026Aug.nc
Original filename: 957f6edfc429ca29e350177ccb5af3fd.nc
Size: 3,666,317,242 bytes.

Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means
Dataset DOI: https://doi.org/10.24381/cds.68d2bb30
Provider: Copernicus Climate Change Service / ECMWF Climate Data Store. Catalogue licence: CC-BY.

Contains 2m temperature (t2m, K) and total precipitation (tp, stored units m). Monthly averaged reanalysis stream moda. Global 0.1-degree grid, 1,801 latitudes by 3,600 longitudes. Exactly 440 consecutive monthly timestamps from January 1990 to August 2026, all labelled 00:00.

The model uses January 2005–December 2014: 120 months sampled at the nearest 0.1-degree grid cell for each lake. Temperature is converted from kelvin to Celsius and averaged with calendar-day weights. For the moda stream, precipitation in metres per day is multiplied by 1,000 and days in the month; annual totals are then averaged over ten complete years. The later precipitation issue affecting September 2022–February 2024 falls outside this window. Documentation: https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation .

The recorded source contains 440 monthly timestamps. The study extracts 291 distinct climate cells for 2,066 lakes; shared cells are considered in the separate grouped validation. The broader source period is not the model predictor window.
