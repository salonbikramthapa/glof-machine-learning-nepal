"""Read-only Streamlit explorer for historical PDGL evaluation results."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='GLOF Research Explorer',page_icon='🏔️',layout='wide')
st.markdown('<style>.block-container{padding-top:2rem;} [data-testid=stSidebar]{background:#edf3f7;} h1{color:#18374b;} </style>',unsafe_allow_html=True)
st.title('GLOF Research Explorer')
st.caption('Machine Learning Approaches · Nepal Himalaya')
st.caption('Salon Bikram Thapa · 17624860 · Machine Learning – Coventry University')
st.info('Historical ICIMOD PDGL membership • Saved held-out results • No training is performed by this app')
@st.cache_data
def load():
    return (pd.read_csv(ROOT/'data/processed/analysis_lakes_climate.csv'),pd.read_csv(ROOT/'results/outer_predictions.csv'),pd.read_csv(ROOT/'results/comparison_with_fold_variability.csv'),json.loads((ROOT/'results/protocol.json').read_text()))
try: lakes,pred,metrics,protocol=load()
except (FileNotFoundError,ValueError) as e:
    st.error(f'Project data are missing or unreadable. Follow the README data setup instructions. {e}');st.stop()
st.sidebar.title('Research workspace')
page=st.sidebar.radio('Navigate',['Home','Dataset','Feature engineering','Model comparison','Evaluation','Interpretation','Recent lake change','Downloads'])
st.sidebar.caption('Unlisted does not mean safe. Scores are not future flood probabilities.')
if page=='Home':
    st.header('Study overview')
    st.write('Explore the data, understand the feature pipeline, compare held-out results and inspect classification errors. Use the navigation on the left to follow the research workflow.')
    for col,label,value in zip(st.columns(4),['Selected lakes','Listed PDGLs','Unlisted lakes','Climate cells'],[len(lakes),int(lakes.historical_pdgl.sum()),int((lakes.historical_pdgl==0).sum()),lakes.climate_cell.nunique()]):col.metric(label,f'{value:,}')
    st.write('The study tests agreement with historical designations in Nepal’s Koshi, Gandaki and Karnali basins. It does not predict flood timing or establish new official PDGLs.')
    st.subheader('Label distribution by basin')
    st.dataframe(lakes.groupby('Basin').historical_pdgl.agg(lakes='size',listed='sum').assign(unlisted=lambda x:x.lakes-x.listed),width='stretch')
    st.subheader('Implemented evaluation')
    st.write('Four classifiers and a majority baseline; five outer stratified folds with three inner tuning folds. A separate three-fold climate-cell grouped sensitivity check keeps each climate cell within a fold. The decision threshold is fixed at 0.5.')
    st.write('Core Random Forest: 8 listed lakes detected, 13 missed and 15 unlisted lakes flagged. This illustrates the detection–alert trade-off, not operational readiness.')
elif page=='Dataset':
    st.header('Dataset explorer')
    st.caption('One row per retained 2015 inventory lake. Filters change this descriptive view only; they never retrain models or change evaluation tables.')
    basin=st.selectbox('Basin filter',['All']+sorted(lakes.Basin.unique().tolist()))
    label=st.selectbox('Historical label filter',['All','Listed PDGL','Unlisted'])
    q=lakes.copy()
    if basin!='All':q=q[q.Basin.eq(basin)]
    if label!='All':q=q[q.historical_pdgl.eq(int(label=='Listed PDGL'))]
    st.metric('Records shown',len(q))
    st.dataframe(q[['GL_ID','Basin','Area','Elevation','Type','historical_pdgl']].rename(columns={'historical_pdgl':'Listed PDGL (1/0)'}),hide_index=True,height=275,width='stretch')
    st.vega_lite_chart(q,{'mark':{'type':'circle','size':35,'opacity':0.65},'encoding':{'x':{'field':'Longitude','type':'quantitative','scale':{'zero':False}},'y':{'field':'Latitude','type':'quantitative','scale':{'zero':False}},'color':{'field':'historical_pdgl','type':'nominal','title':'Listed PDGL (1/0)'}},'height':330},width='stretch')
    st.caption('Coordinate plot in degrees; label 1 means listed and 0 unlisted. This is not a flood inundation map.')
    st.download_button('Download filtered records',q.to_csv(index=False),'filtered_lake_records.csv','text/csv')
    with st.expander('Sources and inclusion rule'):
        st.write('ICIMOD inventory and PDGL membership matched by unique GL_ID. Country Nepal and full polygon containment within the archived geoBoundaries NPL boundary are both required. Four source-Nepal lakes were excluded; no listed Nepal PDGL was removed.')
        st.write('The 2015 inventory supplies the labels and base attributes. ERA5-Land contributes 2005–2014 summaries. GLO 2017–2024 belongs to the separate change extension.')
elif page=='Feature engineering':
    st.header('Feature engineering')
    features=list(dict.fromkeys(f for v in protocol['arms'].values() for f in v))
    units={'Area':'km²','Elevation':'m','Type':'category','Latitude':'degrees N','Longitude':'degrees E','Basin':'category','temp_mean_C_2005_2014':'°C','precip_mean_mm_year_2005_2014':'mm/year'}
    st.dataframe(pd.DataFrame([{'Feature':f,'Units':units[f],'Missing':int(lakes[f].isna().sum()),'Distinct values':lakes[f].nunique()} for f in features]),hide_index=True,width='stretch')
    st.write('Zero missing values in retained predictors. Median / most-frequent imputers remain in the training pipelines as a defined safeguard; they did not fill missing entries in this complete cohort.')
    st.write('LR and KNN: numerical standardisation fitted on training data. Type and basin: training-only one-hot encoding; unseen categories ignored. Tree models: no numerical scaling. IDs, target and hazard ranks: excluded.')
    st.write('Climate: nearest 0.1° cell; 120 monthly observations in 2005–2014. Temperature: Kelvin minus 273.15, weighted by calendar days. Precipitation: daily-equivalent metres × 1,000 × days/month, summed annually then averaged across ten years.')
    st.subheader('Fixed feature sets')
    for arm,cols in protocol['arms'].items():st.write(f"**{arm}:** {', '.join(cols)}")
    with st.expander('Inspect first ten retained records'):
        st.dataframe(lakes[['GL_ID','historical_pdgl']+features].head(10),hide_index=True)
elif page in ['Model comparison','Evaluation','Interpretation']:
    design=st.sidebar.selectbox('Validation',['stratified','cell_grouped'])
    arms=metrics.loc[metrics.validation.eq(design),'arm'].unique().tolist()
    arm=st.sidebar.selectbox('Predictor set',arms)
    m=metrics[(metrics.validation==design)&(metrics.arm==arm)].copy()
    m['accuracy']=(m.tp+m.tn)/(m.tp+m.tn+m.fp+m.fn)
    m['label_rmse']=(1-m.accuracy)**0.5
    if page=='Model comparison':
        st.header('Model comparison')
        st.caption('Pooled held-out metrics on 2,066 lakes; proportions from 0 to 1. Fold F1 SD is descriptive, not a confidence interval.')
        st.dataframe(m[['model','accuracy','precision','recall','f1','balanced_accuracy']].round(3),hide_index=True,width='stretch')
        with st.expander('Confusion counts and fold variability'):
            st.dataframe(m[['model','tp','fp','fn','tn','mean_fold_f1','sd_fold_f1']].round(3),hide_index=True,width='stretch')
        with st.expander('Supplementary RMSE and metric definitions'):
            st.dataframe(m[['model','label_rmse']].round(4),hide_index=True)
            st.write('Label RMSE = sqrt(mean((historical label − predicted class)²)) = sqrt(1 − accuracy). Both labels are 0/1. This duplicates error-rate information and can favour the majority baseline despite zero positive recall. It is not a probability calibration test or a flood-magnitude error.')
            st.write('Precision: listed detections / all positive predictions. Recall: listed detections / all listed lakes. F1: harmonic mean of precision and recall. Balanced accuracy: mean of listed and unlisted recall. Fold SD describes variation, not a confidence interval.')
        with st.expander('Why these models and how settings were chosen'):
            st.write('LR is a regularised linear reference; KNN compares nearby records; a Decision Tree learns nonlinear rules; Random Forest aggregates 100 randomised trees. Majority predicts the most common class. The paper distinguishes direct GLOF applications from general methodological references.')
            st.json(protocol['grids'])
            st.write('Inner-fold positive-class F1 selects settings. LR/DT/RF use balanced class weights; KNN tests uniform and distance-weighted votes. The threshold stays 0.5. No Free Lunch does not forbid comparisons: conclusions are restricted to this sample, metric and validation design.')
        st.bar_chart(m.set_index('model')[['precision','recall','f1']],height=290,stack=False)
        st.download_button('Download selected comparison',m.to_csv(index=False),'model_comparison.csv','text/csv')
        st.warning('A majority prediction achieves about 99% accuracy while detecting no listed lakes. Evaluate recall and alert burden alongside accuracy.')
    elif page=='Evaluation':
        st.header('Evaluation and error analysis')
        model=st.sidebar.selectbox('Model',m.model.tolist(),index=m.model.tolist().index('RandomForest'))
        row=m[m.model.eq(model)].iloc[0]
        st.subheader('Confusion matrix — counts')
        cm=pd.DataFrame([[int(row.tn),int(row.fp)],[int(row.fn),int(row.tp)]],index=['Actual unlisted (0)','Actual listed (1)'],columns=['Predicted unlisted (0)','Predicted listed (1)'])
        st.table(cm)
        st.caption('Rows = historical reference; columns = predictions. Unlisted alerts are not independently verified safe lakes.')
        st.write(f'Listed detected: {int(row.tp)} · Listed missed: {int(row.fn)} · Unlisted flagged: {int(row.fp)}')
        st.subheader('Reference classes across the selected outer folds')
        counts=pred[(pred.validation==design)&(pred.arm==arm)&(pred.model==model)].groupby('fold').y_true.agg(lakes='size',listed='sum').assign(unlisted=lambda x:x.lakes-x.listed)
        st.dataframe(counts,width='stretch')
        q=pred[(pred.validation==design)&(pred.arm==arm)&(pred.model==model)]
        errors=q[q.y_true.ne(q.y_pred)].merge(lakes[['GL_ID','Basin','Latitude','Longitude']],on='GL_ID',validate='one_to_one')
        with st.expander('Inspect disagreement records'):
            st.dataframe(errors[['GL_ID','Basin','Type','y_true','y_pred','fold']],hide_index=True)
        if design=='stratified' and arm=='E1_lake' and model=='RandomForest':st.image(str(ROOT/'results/paper_figures/map.png'),caption='Saved core Random Forest spatial error inspection',width=700)
    else:
        st.header('Held-out feature importance')
        models=[x for x in m.model if x!='Majority'];model=st.sidebar.selectbox('Model',models,index=models.index('RandomForest'))
        imp=pd.read_csv(ROOT/'results/permutation_importance_summary.csv');imp=imp[(imp.validation==design)&(imp.arm==arm)&(imp.model==model)]
        st.dataframe(imp[['feature','mean','std','min','max']].round(4),hide_index=True,width='stretch')
        st.caption('F1 decrease after shuffling an original predictor on held-out folds. The SD spans folds and shuffles; it is not a confidence interval. Importance is model reliance, not causation.')
        if design=='stratified' and arm=='E1_lake' and model=='RandomForest':st.image(str(ROOT/'results/paper_figures/importance_variation.png'),width=550)
elif page=='Recent lake change':
    st.header('Separate recent lake-area extension')
    summary=json.loads((ROOT/'results/recent_change_summary.json').read_text())
    for col,label,value in zip(st.columns(3),['Endpoint-matched lakes','Complete annual panel','Median endpoint change'],[str(summary['matched_endpoint_lakes']),str(summary['complete_eight_year_lakes']),f"{summary['median_percent_change']:.2f}%"]):col.metric(label,value)
    st.subheader('Annual mapped area for the same 855 lakes')
    annual=pd.read_csv(ROOT/'results/recent_lake_complete_panel_totals.csv')
    st.line_chart(annual.set_index('year').rename(columns={'total_area_km2':'Mapped area (km²)'}),height=250)
    with st.expander('Source, filtering and limitations'):st.json(summary)
    st.write('Endpoint comparisons use 1,250 matched lakes. The annual complete panel has 855 lakes. Neither sample supplies new historical PDGL labels. Mapped change is not evidence of climate causation.')
    st.dataframe(pd.read_csv(ROOT/'results/recent_lake_complete_panel_totals.csv'),hide_index=True)

elif page=='Downloads':
    st.header('Results and project files')
    st.write('Download existing evidence for local review. These controls do not train, publish or upload anything. Check source licences before redistributing lake-level data.')
    choices={'All model comparisons':'results/supervisor_revision/all_model_comparisons.csv','Feature quality':'results/supervisor_revision/feature_quality.csv','Basin error summaries':'results/supervisor_revision/errors_by_basin.csv','Frozen experiment protocol':'results/protocol.json'}
    for label,rel in choices.items():
        f=ROOT/rel;st.download_button(label,f.read_bytes(),f.name,key=rel)
    paper=ROOT.parent/'paper/Salon_Bikram_Thapa_17624860.pdf'
    if paper.exists():st.download_button('Research paper PDF',paper.read_bytes(),paper.name,'application/pdf')
    st.subheader('Run and understand the project')
    st.write('Open README.md in the handover for environment setup and notebook order 00–06. Notebook 04 has RUN_TRAINING=False for review. Deliberate reproduction requires a backup, the recorded environment and explicit training activation.')
    st.write('GitHub and YouTube links are not required for the current review and have not been published. The repository preparation and demonstration script are available locally.')
