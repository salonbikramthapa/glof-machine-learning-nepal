from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,geopandas as gpd,xarray as xr
from pyproj import Geod
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
h=Path(__file__).resolve().parents[2];m=h/'GLOF_model';D=h/'outputs/GLOF_MSc_datasets';R=m/'results'
sys.path.insert(0,str(m/'src'))
from verify_results import verify
baseline={str(p.relative_to(h)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [* (R/'models').rglob('*.joblib'),R/'outer_predictions.csv',R/'pooled_metrics.csv',m/'data/processed/analysis_lakes_climate.csv']}
base=gpd.read_file(next((D/'00_ICIMOD_base_2015').rglob('*.shp'))).to_crs(4326);pdgl=gpd.read_file(next((D/'01_ICIMOD_PDGL').rglob('*.shp')));border=gpd.read_file(m/'data/boundary/nepal_adm0.geojson').geometry.union_all()
assert len(base)==3624 and len(pdgl)==47 and base.GL_ID.is_unique and pdgl.GL_ID.is_unique and set(pdgl.GL_ID)<=set(base.GL_ID)
sel=base[base.Country.eq('Nepal')&base.geometry.covered_by(border)].sort_values('GL_ID').reset_index(drop=True)
df=pd.read_csv(m/'data/processed/analysis_lakes_climate.csv').sort_values('GL_ID').reset_index(drop=True)
assert sel.GL_ID.tolist()==df.GL_ID.tolist() and len(sel)==2066
assert np.array_equal(sel.GL_ID.isin(pdgl.GL_ID).astype(int),df.historical_pdgl)
assert set(sel.Basin)=={'Koshi','Gandaki','Karnali'} and sel.is_valid.all()
assert not sel[['Area','Elevation','Type','Latitude','Longitude']].isna().any().any()
for field in ['Area','Elevation','Latitude','Longitude']:assert np.allclose(sel[field],df[field])
geo=Geod(ellps='WGS84');areas=np.array([abs(geo.geometry_area_perimeter(g)[0])/1e6 for g in sel.geometry]);assert ((sel.Area/areas).between(.9,1.1)).all()
assert (df.Type.eq('M(e)')).sum()==301 and df[df.historical_pdgl.eq(1)].Type.eq('M(e)').all()
with xr.open_dataset(next((D/'05_ERA5_Land').glob('*.nc'))) as ds:
 assert 'moda' in ds.attrs['history'] and ds.tp.attrs['units']=='m' and ds.t2m.attrs['units']=='K'
 c=ds[['tp','t2m']].sel(valid_time=slice('2005-01-01','2014-12-31')).sel(latitude=xr.DataArray(df.Latitude.values,dims='lake'),longitude=xr.DataArray(df.Longitude.values,dims='lake'),method='nearest').load()
 assert c.sizes['valid_time']==120 and np.isfinite(c.tp).all() and float(c.tp.min())>=0
 days=c.valid_time.dt.days_in_month
 temp=((c.t2m-273.15)*days).sum('valid_time')/days.sum();prec=(c.tp*1000*days).groupby('valid_time.year').sum('valid_time').mean('year')
 assert np.allclose(temp,df.temp_mean_C_2005_2014) and np.allclose(prec,df.precip_mean_mm_year_2005_2014)
 cell=pd.Series(c.latitude.values).round(1).astype(str)+'_'+pd.Series(c.longitude.values).round(1).astype(str);assert cell.tolist()==df.climate_cell.tolist() and cell.nunique()==291
verification=verify()
fold=pd.read_csv(R/'fold_assignments.csv');settings=json.loads((R/'selected_settings.json').read_text());assert all(s['inner_tuning'] for s in settings if s['model']!='Majority')
inner_checked=0
for design,col in [('stratified','outer_fold'),('cell_grouped','group_fold')]:
 for f in sorted(fold[col].unique()):
  train=df[fold[col]!=f];y=train.historical_pdgl
  cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=17624860+int(f)) if design=='stratified' else StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=17624860+int(f))
  for tr,te in cv.split(train,y,train.climate_cell):
   assert not set(tr)&set(te) and y.iloc[tr].nunique()==2 and y.iloc[te].nunique()==2
   if design=='cell_grouped':assert not set(train.climate_cell.iloc[tr])&set(train.climate_cell.iloc[te])
   inner_checked+=1
rep=pd.read_csv(R/'permutation_importance_repeats.csv');summ=pd.read_csv(R/'permutation_importance_summary.csv')
calc=rep.groupby(['validation','arm','model','feature']).f1_decrease.agg(['mean','std','min','max']).reset_index()
pd.testing.assert_frame_equal(calc,summ,check_exact=False,rtol=1e-10,atol=1e-12)
fm=pd.read_csv(R/'fold_metrics.csv');comp=pd.read_csv(R/'comparison_with_fold_variability.csv')
agg=fm.groupby(['validation','arm','model']).f1.agg(['mean','std']).reset_index().merge(comp,on=['validation','arm','model']);assert np.allclose(agg['mean'],agg.mean_fold_f1) and np.allclose(agg['std'],agg.sd_fold_f1)
g=gpd.read_file(D/'03_GLO_2017_2024/S2_20172024_NTB_GLOID_v1.02.gpkg');assert not g.duplicated(['GLO_ID','AREA_YEAR']).any()
qual=g[g.COUNTRY.eq('Nepal')&g.BASIN.isin(['Koshi','Gandaki','Karnali'])&g.geometry.is_valid&g.to_crs(4326).geometry.covered_by(border)&g.TS_OUTLIER.astype(str).str.lower().eq('false')&g.AREA.gt(0)]
panel=qual.pivot(index='GLO_ID',columns='AREA_YEAR',values='AREA');pair=panel[[2017,2024]].dropna();complete=panel.reindex(columns=range(2017,2025)).dropna()
assert len(pair)==1250 and len(complete)==855
recent=json.loads((R/'recent_change_summary.json').read_text());assert np.isclose(pair[2017].sum(),recent['sum_area_2017_km2']) and np.isclose(pair[2024].sum(),recent['sum_area_2024_km2'])
assert np.isclose(((pair[2024]-pair[2017])/pair[2017]*100).median(),recent['median_percent_change'])
for p,v in baseline.items():assert hashlib.sha256((h/p).read_bytes()).hexdigest()==v
out={'date':'2026-11-04','passed':True,'source_inventory':3624,'source_PDGL':47,'selected_lakes':2066,'selected_PDGL':21,'M_e_lakes':301,'climate_months':120,'climate_cells':291,'inner_partition_designs_reconstructed':inner_checked,'all_nonbaseline_folds_tuned':True,'model_replay':verification,'feature_importance_summary_recomputed':True,'fold_mean_SD_recomputed':True,'recent_endpoints':1250,'recent_complete_panel':855,'protected_files_unchanged':len(baseline),'new_model_training':False,'subset_training_implemented':False,'threshold_selection_implemented':False,'limits':['No independent hazard-label validation','No fresh environment installation','No causal climate inference','Targeted literature review; Gouli full text unavailable']}
(m/'reports/FINAL_AUDIT_2026-11-04.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
