# Analysis table dictionary

This dictionary specifies the actual modelling inputs. Original source metadata are retained beside the input datasets under ../outputs/GLOF_MSc_datasets/.

| Field | Role and interpretation |
|---|---|
| GL_ID | Stable inventory join identifier; never a predictor. |
| historical_pdgl | Target: 1 listed, 0 not listed; not a safe/unsafe truth label. |
| Area | Original mapped lake area, interpreted as km² and corroborated by ellipsoidal polygon calculation. No log transform. |
| Elevation | Original inventory elevation, interpreted as metres; source derived from DEM. |
| Type | Nominal source code; one-hot encoded within training only. No invented ordinal ordering. |
| Latitude / Longitude | Source degrees north/east; used for climate extraction/maps and E4 geography sensitivity only. |
| Basin | Original Koshi/Gandaki/Karnali category; used only in E4 and descriptive reports. |
| temp_mean_C_2005_2014 | Calendar-day-weighted mean monthly 2m temperature in Celsius, nearest ERA5-Land cell. |
| precip_mean_mm_year_2005_2014 | Mean of ten annual precipitation totals in mm/year. moda m/day × 1000 × calendar days before annual sum. |
| climate_latitude / climate_longitude | Coordinates of matched climate grid cell, for audit only. |
| climate_cell | Group identifier for spatial sensitivity; not a predictor. |
| country_nepal / point_inside / polygon_inside / polygon_intersects / included | Boundary-check flags; never predictors. |
| selection_reason | Audit trail; never a predictor. |
| Country / Sub_Basin / Id | Retained source provenance, excluded from all current model feature lists. |

Period: core imagery 2015–2016; historical designation source published 2020; climate ends December 2014. Later lake-change records are analysed separately and do not enter the target table. This dataset is a snapshot, not a per-lake hazard time series.
