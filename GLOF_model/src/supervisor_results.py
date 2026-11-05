"""Derived reporting only: preserve training outputs and recompute summary evidence."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';O=R/'supervisor_revision';O.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/processed/analysis_lakes_climate.csv');pred=pd.read_csv(R/'outer_predictions.csv');metrics=pd.read_csv(R/'comparison_with_fold_variability.csv')
metrics['accuracy']=(metrics.tp+metrics.tn)/(metrics.tp+metrics.tn+metrics.fp+metrics.fn)
metrics['label_rmse']=np.sqrt(1-metrics.accuracy)
for keys,g in pred.groupby(['validation','arm','model']):
 r=metrics[(metrics.validation==keys[0])&(metrics.arm==keys[1])&(metrics.model==keys[2])].iloc[0]
 assert np.isclose(r.label_rmse,np.sqrt(np.mean((g.y_true-g.y_pred)**2)))
metrics.to_csv(O/'all_model_comparisons.csv',index=False)
protocol=json.loads((R/'protocol.json').read_text());features=list(dict.fromkeys(f for cols in protocol['arms'].values() for f in cols))
pd.DataFrame({'feature':features,'missing':[int(df[f].isna().sum()) for f in features],'unique':[df[f].nunique() for f in features]}).to_csv(O/'feature_quality.csv',index=False)
errors=[]
for keys,g in pred.groupby(['validation','arm','model']):
 assert len(g)==len(df) and g.GL_ID.is_unique
 q=g.merge(df[['GL_ID','Basin']],on='GL_ID',validate='one_to_one')
 for basin,b in q.groupby('Basin'):
  errors.append(dict(zip(['validation','arm','model'],keys),Basin=basin,n=len(b),TP=int(((b.y_true==1)&(b.y_pred==1)).sum()),FN=int(((b.y_true==1)&(b.y_pred==0)).sum()),FP=int(((b.y_true==0)&(b.y_pred==1)).sum()),TN=int(((b.y_true==0)&(b.y_pred==0)).sum())))
pd.DataFrame(errors).to_csv(O/'errors_by_basin.csv',index=False)
core=metrics[(metrics.validation=='stratified')&(metrics.arm=='E1_lake')].set_index('model')
fig,axes=plt.subplots(3,2,figsize=(6.6,7.2));names=['Majority','LogisticRegression','KNN','DecisionTree','RandomForest']
for ax,name in zip(axes.ravel(),names):
 r=core.loc[name];a=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int)
 ax.imshow(np.log1p(a),cmap='Blues',vmin=0,vmax=np.log1p(2045))
 for i in range(2):
  for j in range(2):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if a[i,j]>100 else 'black',fontsize=12)
 ax.set_xticks([0,1],['Unlisted','Listed']);ax.set_yticks([0,1],['Unlisted','Listed']);ax.set_title(name,fontsize=10);ax.set_xlabel('Predicted');ax.set_ylabel('Reference')
axes.ravel()[-1].axis('off');fig.tight_layout();fig.savefig(O/'confusion_matrices.png',dpi=250);plt.close(fig)
print('Derived comparison, feature quality, basin errors and confusion matrices saved. No training.')
