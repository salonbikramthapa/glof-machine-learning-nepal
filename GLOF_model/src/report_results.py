"""Generate transparent figures and a results summary from saved outputs."""
from pathlib import Path
import os,json
import numpy as np
import pandas as pd
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'verification/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
OUT=ROOT/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
MODELS=['Majority','LogisticRegression','KNN','DecisionTree','RandomForest']
LABELS=['Majority','Logistic regression','KNN','Decision tree','Random forest']
def run():
    metrics=pd.read_csv(OUT/'pooled_metrics.csv');pred=pd.read_csv(OUT/'outer_predictions.csv');df=pd.read_csv(ROOT/'data/processed/analysis_lakes_climate.csv')
    folds=pd.read_csv(OUT/'fold_metrics.csv')
    # Pooled and mean-fold results are different estimands; show both explicitly.
    mean=folds.groupby(['validation','arm','model']).agg(mean_fold_f1=('f1','mean'),sd_fold_f1=('f1','std'),mean_fold_recall=('recall','mean'),sd_fold_recall=('recall','std')).reset_index()
    combined=metrics.merge(mean,on=['validation','arm','model']);combined.to_csv(OUT/'comparison_with_fold_variability.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,validation in zip(axes,['stratified','cell_grouped']):
        for arm,marker in [('E1_lake','o'),('E3_climate','s')]:
            g=metrics[(metrics.validation==validation)&(metrics.arm==arm)].set_index('model').loc[MODELS]
            ax.plot(range(5),g.f1,marker=marker,label='Lake only' if arm=='E1_lake' else 'Lake plus climate')
        ax.set_xticks(range(5),LABELS,rotation=30,ha='right');ax.set_ylim(0,.55);ax.set_ylabel('Pooled held-out F1');ax.set_title('Five stratified folds' if validation=='stratified' else 'Three climate-cell grouped folds');ax.legend()
    fig.tight_layout();fig.savefig(FIG/'model_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(14,3))
    for ax,name,label in zip(axes,MODELS,LABELS):
        r=metrics[(metrics.validation=='stratified')&(metrics.arm=='E1_lake')&(metrics.model==name)].iloc[0]
        ConfusionMatrixDisplay(np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int),display_labels=['Unlisted','PDGL']).plot(ax=ax,colorbar=False,values_format='d',cmap='Blues')
        ax.set_title(label)
    fig.tight_layout();fig.savefig(FIG/'core_confusion_matrices.png',dpi=180);plt.close(fig)
    importance=pd.read_csv(OUT/'permutation_importance_repeats.csv')
    core=importance[(importance.validation=='stratified')&(importance.arm=='E1_lake')]
    fig,axes=plt.subplots(1,4,figsize=(12,3.5))
    for ax,name in zip(axes,MODELS[1:]):
        per_fold=core[core.model==name].groupby(['feature','fold']).f1_decrease.mean().unstack('fold')
        per_fold=per_fold.reindex(['Area','Elevation','Type'])
        ax.bar(range(3),per_fold.mean(axis=1),color='#3d6980')
        for i,(_,row) in enumerate(per_fold.iterrows()):ax.scatter(np.full(len(row),i),row,color='#be4b37',s=15,zorder=3)
        ax.axhline(0,color='grey',linewidth=.7);ax.set_xticks(range(3),['Area','Elevation','Type'],rotation=30);ax.set_title(name);ax.set_ylabel('Held-out F1 decrease')
    fig.tight_layout();fig.savefig(FIG/'core_permutation_importance.png',dpi=180);plt.close(fig)
    border=gpd.read_file(ROOT/'data/boundary/nepal_adm0.geojson').to_crs(4326)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    categories=[('True negative','#b8c5cc',5),('Unlisted flagged','#e08e31',20),('PDGL missed','#b32435',35),('PDGL detected','#287c46',35)]
    for ax,name in zip(axes.flat,MODELS[1:]):
        g=pred[(pred.validation=='stratified')&(pred.arm=='E1_lake')&(pred.model==name)].merge(df[['GL_ID','Latitude','Longitude']],on='GL_ID',validate='one_to_one')
        g['error_category']=np.select([(g.y_true==0)&(g.y_pred==0),(g.y_true==0)&(g.y_pred==1),(g.y_true==1)&(g.y_pred==0)],['True negative','Unlisted flagged','PDGL missed'],default='PDGL detected')
        border.boundary.plot(ax=ax,color='#666666',linewidth=.5)
        for cat,color,size in categories:
            q=g[g.error_category==cat];ax.scatter(q.Longitude,q.Latitude,c=color,s=size,label=cat,alpha=.8)
        ax.set_title(name);ax.set(xlim=(80,88.4),ylim=(26.3,30.8),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
        if name=='LogisticRegression':ax.legend(fontsize=7,loc='lower left')
        g.to_csv(OUT/f'core_error_map_{name}.csv',index=False)
    fig.suptitle('Held-out historical designation errors — not a flood forecast',fontsize=12);fig.tight_layout();fig.savefig(FIG/'held_out_error_maps.png',dpi=180);plt.close(fig)
    recent=pd.read_csv(OUT/'recent_lake_complete_panel_totals.csv')
    fig,ax=plt.subplots(figsize=(7,3));ax.plot(recent.year,recent.total_area_km2,marker='o');ax.set(xlabel='Year',ylabel='Mapped lake area (km²)',title='Fixed panel of 855 qualifying lakes — descriptive extension');fig.tight_layout();fig.savefig(FIG/'recent_fixed_panel.png',dpi=180);plt.close(fig)
    paired=metrics[metrics.validation.eq('stratified')].pivot(index='model',columns='arm',values='f1');paired['climate_minus_lake_f1']=paired.E3_climate-paired.E1_lake;paired['without_type_minus_lake_f1']=paired.E2_no_type-paired.E1_lake;paired.to_csv(OUT/'paired_arm_f1_differences.csv')
    return combined
if __name__=='__main__':run()
