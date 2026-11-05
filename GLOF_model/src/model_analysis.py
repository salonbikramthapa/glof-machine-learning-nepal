"""Nested evaluation: preprocessing and tuning inside training partitions only.
Never use pooled outer scores to tune parameters, select features, or alter thresholds.
"""
from pathlib import Path
import os,json,hashlib,time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score,confusion_matrix,make_scorer
from sklearn.inspection import permutation_importance
import joblib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
SEED=17624860
ARMS={'E1_lake':['Area','Elevation','Type'],'E2_no_type':['Area','Elevation'],'E3_climate':['Area','Elevation','Type','temp_mean_C_2005_2014','precip_mean_mm_year_2005_2014'],'E4_geography':['Area','Elevation','Type','Latitude','Longitude','Basin']}
GRIDS={'Majority':{},'LogisticRegression':{'model__C':[.1,1.,10.]},'KNN':{'model__n_neighbors':[3,7,15],'model__weights':['uniform','distance']},'DecisionTree':{'model__max_depth':[2,4,None],'model__min_samples_leaf':[1,5]},'RandomForest':{'model__max_depth':[4,None],'model__min_samples_leaf':[1,5]}}
def pipeline(name,features):
    numeric=[x for x in features if x not in ['Type','Basin']];categorical=[x for x in features if x in ['Type','Basin']]
    steps=[('impute',SimpleImputer(strategy='median'))]
    if name in ['LogisticRegression','KNN']:steps.append(('scale',StandardScaler()))
    transforms=[('numeric',Pipeline(steps),numeric)]
    if categorical:transforms.append(('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical))
    models={'Majority':DummyClassifier(strategy='most_frequent'),'LogisticRegression':LogisticRegression(class_weight='balanced',max_iter=3000,random_state=SEED),'KNN':KNeighborsClassifier(),'DecisionTree':DecisionTreeClassifier(class_weight='balanced',random_state=SEED),'RandomForest':RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=1)}
    return Pipeline([('preprocess',ColumnTransformer(transforms,remainder='drop')),('model',models[name])])
def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'f1':f1_score(y,p,zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,p),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
def threshold_f1(estimator, X, y):
    return f1_score(y, (estimator.predict_proba(X)[:,1]>=.5).astype(int), zero_division=0)

def valid_splits(splits,y,groups=None,minimum=3):
    for tr,te in splits:
        if len(np.intersect1d(tr,te)) or y.iloc[te].sum()<minimum or y.iloc[tr].nunique()!=2 or y.iloc[te].nunique()!=2:return False
        if groups is not None and set(groups.iloc[tr])&set(groups.iloc[te]):return False
    return True

def freeze():
    source=ROOT/'data/processed/analysis_lakes_climate.csv';df=pd.read_csv(source).sort_values('GL_ID').reset_index(drop=True);y=df.historical_pdgl
    assert df.GL_ID.is_unique and y.nunique()==2
    splits=None
    for n in [5,3]:
        candidate=list(StratifiedKFold(n_splits=n,shuffle=True,random_state=SEED).split(df,y))
        if valid_splits(candidate,y):splits=candidate;break
    assert splits is not None,'Insufficient positives for proposal minimum.'
    df['outer_fold']=-1
    for i,(_,te) in enumerate(splits):df.loc[te,'outer_fold']=i
    assert (df.outer_fold>=0).all()
    # A priori 3-fold climate-cell grouping sensitivity; not a geographic transfer test.
    groups=df.climate_cell
    gs=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(df,y,groups))
    group_ok=valid_splits(gs,y,groups)
    df['group_fold']=-1
    if group_ok:
        for i,(_,te) in enumerate(gs):df.loc[te,'group_fold']=i
    protocol={'seed':SEED,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n':len(df),'positives':int(y.sum()),'arms':ARMS,'models':list(GRIDS),'grids':GRIDS,'outer_folds':len(splits),'inner_folds':3,'threshold':.5,'scorer':'positive-class F1, zero_division=0','class_weight':'balanced for LR/DT/RF; KNN vote weights only; no synthetic resampling','permutation':'10 shuffles of each original column in outer test partition; F1 decrease; descriptive variability only','geography_change':'E4 adds coordinates and basin; E1 is the removal comparison required by proposal','type_change':'E2 and fixed M(e) rule are blueprint refinements, not original proposal models','group_sensitivity':'3 outer/inner climate-cell grouped folds when feasible; no adjacent-cell buffer and no claim of basin transfer','group_feasible':bool(group_ok),'interpretation':'retrospective historical designation classification; scores are not flood probabilities','boundary_rule':'source Nepal AND full polygon covered by research ADM0; excluded disagreements logged before modelling','software':'../requirements-lock.txt'}
    target=OUT/'protocol.json'
    if target.exists():assert json.loads(target.read_text())==protocol,'Protocol changed. Create a separate experiment rather than overwrite.'
    else:target.write_text(json.dumps(protocol,indent=2))
    df[['GL_ID','historical_pdgl','climate_cell','outer_fold','group_fold']].to_csv(OUT/'fold_assignments.csv',index=False)
    df.groupby('outer_fold').historical_pdgl.agg(['count','sum']).to_csv(OUT/'fold_class_counts.csv')
    print('Frozen protocol:',len(df),'lakes;',len(splits),'outer folds; grouped feasible:',group_ok,flush=True)
    return df,protocol

def run():
    df,protocol=freeze();y=df.historical_pdgl
    predrows=[];scores=[];settings=[];importances=[];inner_records=[]
    scorer=threshold_f1
    designs=[('stratified',list(ARMS))]
    if protocol['group_feasible']:designs.append(('cell_grouped',['E1_lake','E3_climate']))
    for validation,arms in designs:
        foldcol='outer_fold' if validation=='stratified' else 'group_fold'
        for arm in arms:
            features=ARMS[arm];X=df[features]
            for name in GRIDS:
                for fold in sorted(df[foldcol].unique()):
                    tr=np.flatnonzero(df[foldcol]!=fold);te=np.flatnonzero(df[foldcol]==fold)
                    yt=y.iloc[tr];groups=df.climate_cell.iloc[tr]
                    if validation=='stratified':inner=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED+int(fold)).split(X.iloc[tr],yt))
                    else:inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+int(fold)).split(X.iloc[tr],yt,groups))
                    feasible=valid_splits(inner,yt,groups if validation=='cell_grouped' else None,minimum=1)
                    for j,(a,b) in enumerate(inner):
                        inner_records.append({'validation':validation,'arm':arm,'model':name,'outer_fold':int(fold),'inner_fold':j,'train_n':len(a),'test_n':len(b),'train_positive':int(yt.iloc[a].sum()),'test_positive':int(yt.iloc[b].sum()),'group_overlap':len(set(groups.iloc[a])&set(groups.iloc[b])) if validation=='cell_grouped' else None})
                    pipe=pipeline(name,features)
                    if name!='Majority' and feasible:
                        search=GridSearchCV(pipe,GRIDS[name],scoring=scorer,cv=inner,n_jobs=1,error_score='raise',refit=True)
                        search.fit(X.iloc[tr],yt);best=search.best_estimator_;params=search.best_params_;inner_f1=float(search.best_score_)
                    else:
                        # Prespecified middle/small settings, not chosen on outer tests.
                        params={'model__C':1.} if name=='LogisticRegression' else {'model__n_neighbors':7,'model__weights':'uniform'} if name=='KNN' else {'model__max_depth':4,'model__min_samples_leaf':5} if name in ['DecisionTree','RandomForest'] else {}
                        best=pipe.set_params(**params).fit(X.iloc[tr],yt);inner_f1=None
                    prob=best.predict_proba(X.iloc[te])[:,1];p=(prob>=.5).astype(int)
                    key={'validation':validation,'arm':arm,'model':name,'fold':int(fold)}
                    scores.append(dict(key,train_n=len(tr),test_n=len(te),train_positive=int(yt.sum()),test_positive=int(y.iloc[te].sum()),**metrics(y.iloc[te],p)))
                    settings.append(dict(key,parameters=params,inner_f1=inner_f1,inner_tuning=bool(feasible and name!='Majority')))
                    for j,idx in enumerate(te):predrows.append(dict(key,GL_ID=df.GL_ID.iloc[idx],y_true=int(y.iloc[idx]),y_pred=int(p[j]),score=float(prob[j]),Type=df.Type.iloc[idx]))
                    if name!='Majority':
                        importance=permutation_importance(best,X.iloc[te],y.iloc[te],scoring=scorer,n_repeats=10,random_state=SEED+int(fold),n_jobs=1)
                        for j,feature in enumerate(features):
                            for repeat,val in enumerate(importance.importances[j]):importances.append(dict(key,feature=feature,repeat=repeat,f1_decrease=float(val)))
                    folder=OUT/'models'/validation/arm;folder.mkdir(parents=True,exist_ok=True)
                    joblib.dump(best,folder/f'{name}_fold{fold}.joblib',compress=3)
                    print(validation,arm,name,'fold',fold,'done',flush=True)
                # Checkpoint each completed model so a failed later run remains inspectable.
                pd.DataFrame(predrows).to_csv(OUT/'outer_predictions.csv',index=False)
                pd.DataFrame(scores).to_csv(OUT/'fold_metrics.csv',index=False)
    predictions=pd.DataFrame(predrows)
    assert not predictions.duplicated(['validation','arm','model','GL_ID']).any()
    pooled=[]
    for key,g in predictions.groupby(['validation','arm','model']):
        assert len(g)==len(df)
        pooled.append(dict(zip(['validation','arm','model'],key),**metrics(g.y_true,g.y_pred)))
    pd.DataFrame(pooled).to_csv(OUT/'pooled_metrics.csv',index=False)
    pd.DataFrame(scores).groupby(['validation','arm','model'])[['precision','recall','f1','balanced_accuracy']].agg(['mean','std','min','max']).to_csv(OUT/'fold_metric_summary.csv')
    (OUT/'selected_settings.json').write_text(json.dumps(settings,indent=2))
    pd.DataFrame(inner_records).to_csv(OUT/'inner_fold_checks.csv',index=False)
    pd.DataFrame(importances).to_csv(OUT/'permutation_importance_repeats.csv',index=False)
    pd.DataFrame(importances).groupby(['validation','arm','model','feature']).f1_decrease.agg(['mean','std','min','max']).to_csv(OUT/'permutation_importance_summary.csv')
    rule=df.Type.eq('M(e)').astype(int)
    (OUT/'type_rule.json').write_text(json.dumps({'note':'Fixed descriptive benchmark motivated by prior audit, not learned or independently validated',**metrics(y,rule)},indent=2))
    within=[]
    for key,g in predictions[predictions.Type.eq('M(e)')].groupby(['validation','arm','model']):within.append(dict(zip(['validation','arm','model'],key),n=len(g),**metrics(g.y_true,g.y_pred)))
    pd.DataFrame(within).to_csv(OUT/'within_type_metrics.csv',index=False)
    print('All specified evaluations complete.',flush=True)
if __name__=='__main__':run()
