"""Deterministic, label-independent boundary selection and dated climate features."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'outputs/GLOF_MSc_datasets'
OUT=ROOT/'data/processed'; REPORT=ROOT/'reports'
def prepare():
    base=gpd.read_file(next((DATA/'00_ICIMOD_base_2015').rglob('*.shp'))).to_crs(4326)
    pdgl=gpd.read_file(next((DATA/'01_ICIMOD_PDGL').rglob('*.shp')))
    assert base.GL_ID.is_unique and pdgl.GL_ID.is_unique
    assert set(pdgl.GL_ID)<=set(base.GL_ID)
    border=gpd.read_file(ROOT/'data/boundary/nepal_adm0.geojson').to_crs(4326)
    assert border.is_valid.all() and base.is_valid.all()
    union=border.geometry.union_all()
    base['country_nepal']=base.Country.eq('Nepal')
    base['point_inside']=gpd.GeoSeries(gpd.points_from_xy(base.Longitude,base.Latitude),crs=4326).covered_by(union)
    base['polygon_inside']=base.geometry.covered_by(union)
    base['polygon_intersects']=base.intersects(union)
    base['historical_pdgl']=base.GL_ID.isin(pdgl.GL_ID).astype(int)
    base['included']=base.country_nepal & base.polygon_inside
    base['selection_reason']=np.select([base.included,base.country_nepal & ~base.polygon_inside,~base.country_nepal & base.polygon_intersects],['country_and_full_polygon_agree','exclude_border_or_spatial_disagreement','exclude_country_disagreement'],default='outside_study_scope')
    base.drop(columns='geometry').to_csv(REPORT/'boundary_all_records.csv',index=False)
    review=base[(base.country_nepal!=base.point_inside) | (base.country_nepal & ~base.polygon_inside) | (~base.country_nepal & base.polygon_intersects)]
    review.drop(columns='geometry').to_csv(REPORT/'boundary_review.csv',index=False)
    lakes=base[base.included].sort_values('GL_ID').copy().reset_index(drop=True)
    assert lakes[['Area','Elevation','Type','Latitude','Longitude']].notna().all().all()
    assert np.isfinite(lakes[['Area','Elevation','Latitude','Longitude']]).all().all()
    assert (lakes.Area>0).all() and (lakes.Elevation>0).all()
    # All shared PDGL/base predictor fields must agree for included positives.
    paired=lakes.merge(pdgl.drop(columns='geometry'),on='GL_ID',suffixes=('_base','_pdgl'),validate='one_to_one')
    for c in ['Country','Type','Basin','Area','Elevation','Latitude','Longitude']:
        assert paired[c+'_base'].equals(paired[c+'_pdgl']), c
    lakes.to_file(OUT/'analysis_lakes.gpkg',layer='lakes',driver='GPKG',mode='w')
    lakes.drop(columns='geometry').to_csv(OUT/'analysis_lakes.csv',index=False)
    summary={'primary_rule':'Country == Nepal AND full polygon covered by geoBoundaries NPL ADM0; exclude disagreements without editing source country','boundary_year':2019,'boundary_id':'NPL-ADM0-63295607','source_country_count':int(base.country_nepal.sum()),'point_inside_count':int(base.point_inside.sum()),'full_polygon_inside_count':int(base.polygon_inside.sum()),'analysis_count':len(lakes),'positives':int(lakes.historical_pdgl.sum()),'excluded_country_nepal':int((base.country_nepal & ~base.included).sum()),'excluded_positive_country_nepal':int(base.loc[base.country_nepal & ~base.included,'historical_pdgl'].sum()),'review_records':len(review),'limitations':'Later research boundary; full-containment rule excludes crossing lakes. Supervisor should review this explicit operational choice; no claim to adjudicate borders.'}
    (REPORT/'boundary_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return lakes

def climate():
    lakes=pd.read_csv(OUT/'analysis_lakes.csv')
    source=next((DATA/'05_ERA5_Land').glob('*.nc'))
    with xr.open_dataset(source) as ds:
        assert 'moda' in ds.attrs['history']
        assert ds.tp.attrs['units']=='m' and ds.t2m.attrs['units']=='K'
        sub=ds[['t2m','tp']].sel(valid_time=slice('2005-01-01','2014-12-31'),latitude=slice(lakes.Latitude.max()+.2,lakes.Latitude.min()-.2),longitude=slice(lakes.Longitude.min()-.2,lakes.Longitude.max()+.2)).load()
        dates=pd.DatetimeIndex(sub.valid_time.values)
        assert dates.equals(pd.date_range('2005-01-01','2014-12-01',freq='MS'))
        cells=sub.sel(latitude=xr.DataArray(lakes.Latitude.to_numpy(),dims='lake'),longitude=xr.DataArray(lakes.Longitude.to_numpy(),dims='lake'),method='nearest')
        assert np.isfinite(cells.t2m).all() and np.isfinite(cells.tp).all() and float(cells.tp.min())>=0
        days=cells.valid_time.dt.days_in_month
        lakes['temp_mean_C_2005_2014']=((cells.t2m-273.15)*days).sum('valid_time').values/float(days.sum())
        monthly=cells.tp*1000*days
        annual=monthly.groupby('valid_time.year').sum('valid_time')
        lakes['precip_mean_mm_year_2005_2014']=annual.mean('year').values
        lakes['climate_latitude']=cells.latitude.values;lakes['climate_longitude']=cells.longitude.values
        lakes['climate_cell']=lakes.climate_latitude.round(1).astype(str)+'_'+lakes.climate_longitude.round(1).astype(str)
        cells.to_netcdf(OUT/'climate_lake_months_2005_2014.nc')
        summary={'period':'2005-01 to 2014-12','months':len(dates),'lakes':len(lakes),'unique_cells':int(lakes.climate_cell.nunique()),'temperature':'day-weighted monthly means, K minus 273.15','precipitation':'moda m/day x 1000 x calendar days; sum each year then mean 10 annual totals','grid':'0.1 degree nearest cell; shared cells are not independent climate observations','finite':True,'source':str(source.relative_to(ROOT.parent)),'source_bytes':source.stat().st_size,'time_alignment':'ICIMOD saved XML documents 2015–2016 imagery and PDGL assessment based on 2015 inventory; summaries stop December 2014','known_issue':'September 2022–February 2024 accumulated precipitation not used; no recent precipitation trend claimed','documentation':'https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation','citation':'https://doi.org/10.24381/cds.68d2bb30'}
    lakes.to_csv(OUT/'analysis_lakes_climate.csv',index=False)
    (REPORT/'climate_extraction.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return lakes
if __name__=='__main__':prepare();climate()
