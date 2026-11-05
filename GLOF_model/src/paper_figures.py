from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'verification/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd,numpy as np
p=Path(__file__).resolve().parents[1]/'results';f=p/'paper_figures';f.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'savefig.dpi':250})
fig,ax=plt.subplots(figsize=(3.3,3.5));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
texts=['ICIMOD inventories + boundary checks\n2,066 lakes; 21 historical PDGLs','Fixed feature sets E1–E4\nClimate inputs precede inventory imagery','Five outer folds: hold one fold aside\nKeep outer test data out of all tuning','Inside outer training data: 3 inner folds\nFit preprocessing + select settings by F1','Refit on outer training data\nPredict held-out fold at threshold 0.5','Pool held-out predictions; summarise\nMetrics, errors and fold importance']
ys=np.linspace(.91,.09,6)
for i,(y,t) in enumerate(zip(ys,texts)):
 ax.add_patch(FancyBboxPatch((.025,y-.06),.95,.12,boxstyle='round,pad=.008',facecolor='#edf3f7',edgecolor='#385b73',linewidth=.7))
 ax.text(.5,y,t,ha='center',va='center',fontsize=7.5)
 if i<5:ax.annotate('',xy=(.5,ys[i+1]+.069),xytext=(.5,y-.068),arrowprops={'arrowstyle':'->','lw':.8,'color':'#385b73'})
fig.subplots_adjust(left=0,right=1,top=1,bottom=0);fig.savefig(f/'workflow.png');plt.close(fig)
a=pd.read_csv(p/'permutation_importance_repeats.csv');a=a[(a.validation=='stratified')&(a.arm=='E1_lake')&(a.model=='RandomForest')]
fig,ax=plt.subplots(figsize=(3.3,2.7))
for i,feature in enumerate(['Area','Elevation','Type']):
 vals=a[a.feature.eq(feature)].groupby('fold').f1_decrease.mean()
 ax.scatter(vals,np.arange(5)*.045+i-.09,c='#427a9c',s=17,label='Fold mean' if i==0 else None)
 ax.scatter([vals.mean()],[i],marker='D',c='#bd6030',s=30,label='Across-fold mean' if i==0 else None,zorder=3)
ax.axvline(0,color='#777',lw=.7);ax.set_yticks(range(3),['Area','Elevation','Type']);ax.invert_yaxis();ax.set_xlabel('Held-out F1 decrease');ax.legend(fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=1);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(f/'importance_variation.png');plt.close(fig)
