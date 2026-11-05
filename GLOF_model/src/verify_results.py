"""Check saved results against fold definitions and persisted estimators."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd
import numpy as np
import joblib
from model_analysis import ROOT,ARMS,metrics

def verify():
    out=ROOT/'results';df=pd.read_csv(ROOT/'data/processed/analysis_lakes_climate.csv').sort_values('GL_ID').reset_index(drop=True)
    p=pd.read_csv(out/'outer_predictions.csv');folds=pd.read_csv(out/'fold_assignments.csv');protocol=json.loads((out/'protocol.json').read_text())
    assert hashlib.sha256((ROOT/'data/processed/analysis_lakes_climate.csv').read_bytes()).hexdigest()==protocol['input_sha256']
    assert df.GL_ID.equals(folds.GL_ID)
    assert not p.duplicated(['validation','arm','model','GL_ID']).any()
    pooled=pd.read_csv(out/'pooled_metrics.csv')
    replayed=0
    for (validation,arm,model,fold),g in p.groupby(['validation','arm','model','fold']):
        foldcol='outer_fold' if validation=='stratified' else 'group_fold'
        te=folds[foldcol].eq(fold);tr=~te
        assert set(g.GL_ID)==set(folds.loc[te,'GL_ID'])
        assert df.loc[te,'historical_pdgl'].sum()>=3
        if validation=='cell_grouped':assert not set(folds.loc[tr,'climate_cell'])&set(folds.loc[te,'climate_cell'])
        pipe=joblib.load(out/'models'/validation/arm/f'{model}_fold{fold}.joblib')
        rows=df.set_index('GL_ID').loc[g.GL_ID]
        actual=pipe.predict_proba(rows[ARMS[arm]])[:,1]
        assert np.allclose(actual,g.score,rtol=0,atol=1e-14)
        assert np.array_equal((actual>=.5).astype(int),g.y_pred)
        assert np.array_equal(rows.historical_pdgl,g.y_true)
        # Imputation/scaling statistics must match this outer training subset.
        pre=pipe.named_steps['preprocess'];numeric=[x for x in ARMS[arm] if x not in ['Type','Basin']]
        imp=pre.named_transformers_['numeric'].named_steps['impute']
        assert np.allclose(imp.statistics_,df.loc[tr,numeric].median().values)
        if model in ['LogisticRegression','KNN']:
            scaler=pre.named_transformers_['numeric'].named_steps['scale'];assert np.allclose(scaler.mean_,df.loc[tr,numeric].mean().values)
        replayed+=1
    for (validation,arm,model),g in p.groupby(['validation','arm','model']):
        assert set(g.GL_ID)==set(df.GL_ID)
        saved=pooled[(pooled.validation==validation)&(pooled.arm==arm)&(pooled.model==model)].iloc[0]
        for k,v in metrics(g.y_true,g.y_pred).items():assert np.isclose(saved[k],v)
    inner=pd.read_csv(out/'inner_fold_checks.csv');assert (inner.train_positive>0).all() and (inner.test_positive>0).all();assert (inner.loc[inner.validation.eq('cell_grouped'),'group_overlap']==0).all()
    assert not any('Rank' in f or 'historical_pdgl' in f or 'GL_ID' in f for fs in ARMS.values() for f in fs)
    summary={'passed':True,'replayed_outer_estimators':replayed,'prediction_rows':len(p),'cohort_sha256':protocol['input_sha256'],'checks':['one held-out prediction per lake/model/arm/design','prediction replay from saved fold estimators','outer training medians and scale means','all primary outer tests contain at least three positives','grouped outer and inner cell disjointness','pooled confusion counts and metrics recalculated','predictor allowlists exclude target and IDs'],'limitation':'Computational checks do not independently validate source labels, boundary accuracy, or external generalisation.'}
    (out/'VERIFICATION.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':verify()
