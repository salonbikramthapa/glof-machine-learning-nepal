"""Exercise saved-results views without fitting any model."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
assert not at.exception
checks=[]
for page in ['Home','Dataset','Feature engineering','Model comparison','Evaluation','Interpretation','Recent lake change','Downloads']:
 at.sidebar.radio[0].set_value(page).run();assert not at.exception,(page,at.exception)
 if page in ['Model comparison','Evaluation','Interpretation']:
  for design in ['stratified','cell_grouped']:
   at.selectbox[0].set_value(design).run();assert not at.exception
   for arm in at.selectbox[1].options:
    at.selectbox[1].set_value(arm).run();assert not at.exception
    if len(at.selectbox)>2:
     for model in at.selectbox[2].options:
      at.selectbox[2].set_value(model).run();assert not at.exception
      checks.append([page,design,arm,model])
    else:checks.append([page,design,arm])
 elif page=='Dataset':
  for basin in at.selectbox[0].options:
   at.selectbox[0].set_value(basin).run();assert not at.exception
   for label in at.selectbox[1].options:
    at.selectbox[1].set_value(label).run();assert not at.exception
    checks.append([page,basin,label])
 else:checks.append([page])
(ROOT/'reports/STREAMLIT_TESTS.json').write_text(json.dumps({'passed':True,'tested_views':checks,'no_training':True,'method':'Streamlit AppTest; every available selector combination'},indent=2))
print('PASS',len(checks),'view combinations')
