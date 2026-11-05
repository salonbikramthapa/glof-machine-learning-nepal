"""Separate descriptive extension; never enters the historical ML table."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[1]
def run():
    source=ROOT.parent/'outputs/GLOF_MSc_datasets/03_GLO_2017_2024/S2_20172024_NTB_GLOID_v1.02.gpkg'
    g=gpd.read_file(source)
    print(g.columns.tolist(),flush=True)
    assert not g.duplicated(['GLO_ID','AREA_YEAR']).any()
    b=gpd.read_file(ROOT/'data/boundary/nepal_adm0.geojson').to_crs(g.crs).geometry.union_all()
    eligible=(g.COUNTRY.eq('Nepal') & g.BASIN.isin(['Koshi','Gandaki','Karnali']) & g.geometry.is_valid & g.geometry.covered_by(b) & g.TS_OUTLIER.astype(str).str.lower().eq('false') & g.AREA.gt(0))
    selected=g[eligible].copy()
    endpoints=selected[selected.AREA_YEAR.isin([2017,2024])]
    paired=endpoints.pivot(index='GLO_ID',columns='AREA_YEAR',values='AREA').dropna()
    paired.columns=['area_2017_km2','area_2024_km2']
    paired['change_km2']=paired.area_2024_km2-paired.area_2017_km2
    paired['change_percent']=100*paired.change_km2/paired.area_2017_km2
    paired.to_csv(ROOT/'results/recent_lake_paired_change.csv')
    # Fixed matched panel, not changing yearly sample. 8 valid years required.
    panel=selected.pivot(index='GLO_ID',columns='AREA_YEAR',values='AREA').reindex(columns=list(range(2017,2025))).dropna()
    panel.sum(axis=0).rename('total_area_km2').to_csv(ROOT/'results/recent_lake_complete_panel_totals.csv',index_label='year')
    summary={'separate_extension':True,'source':str(source.relative_to(ROOT.parent)),'quality_rule':'Nepal source country, permitted basin, valid full polygon inside same research boundary, TS_OUTLIER explicitly FALSE; unknown flags excluded','all_source_rows':len(g),'eligible_rows':len(selected),'matched_endpoint_lakes':len(paired),'complete_eight_year_lakes':len(panel),'sum_area_2017_km2':float(paired.area_2017_km2.sum()),'sum_area_2024_km2':float(paired.area_2024_km2.sum()),'sum_change_km2':float(paired.change_km2.sum()),'median_percent_change':float(paired.change_percent.median()),'limitations':'Matched surviving mapped lakes only; selection and sensor/composite effects. Provider area uncertainties not propagated; descriptive point estimates, not significance. No inference about official PDGL growth, lake formation frequency or climate causation.'}
    (ROOT/'results/recent_change_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return summary
if __name__=='__main__':run()
