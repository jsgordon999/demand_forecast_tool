import time
import os  
import numpy as np
import pandas as pd
import json
import streamlit as st
from NN_demand_forecast import NNForecaster
import matplotlib.pyplot as plt
import lightgbm as lgb

st.set_page_config(page_title='Demand Forecasting', layout='wide')

# ---------- These will cache items for faster re-loading of page ----------
@st.cache_data
def load_data():
    data = np.load('prediction_time_data.npz')
    with open('product_stats.json') as f:
        d = json.load(f)
    product_stats = {int(k): tuple(v) for k, v in d.items()}

    return {
        'feat_cols': data['feat_cols'],
        'store_ids': data['store_ids'],
        'sku_ids': data['sku_ids'],
        'scaler_mean': data['scaler_mean'],
        'scaler_scale': data['scaler_scale'],
        'product_stats': product_stats
    }

@st.cache_resource
def load_models():
    nn = NNForecaster(
        n_stores=76, 
        n_skus=28,
        hidden=512,
        depth=3,
        lr=1e-3,
        decayevery=50,
        decayrate=0.5
    )
    nn.load('NN_forecaster.keras')
    booster = lgb.Booster(model_file='lgbm_model.txt')
    return nn, booster

@st.cache_data
def nn_predict_cached(store_ids, sku_ids, Xcont):
    return nn.predict_units(store_ids, sku_ids, Xcont, return_log=False)

@st.cache_data 
def lgbm_predict_cached(store_ids, sku_ids, Xcont):
    store_ids = store_ids.reshape(-1,1)
    sku_ids = sku_ids.reshape(-1,1)
    X_pred = np.concatenate([store_ids, sku_ids, Xcont], axis=1)
    return np.expm1(booster.predict(X_pred))

# Loading data 
data = load_data()
nn, booster = load_models()

feat_cols = data['feat_cols']
store_ids = data['store_ids']
sku_ids = data['sku_ids']
scaler_mean = data['scaler_mean']
scaler_scale = data['scaler_scale']
product_stats = data['product_stats']


st.title('Demand Forecasting')

# ---------- Problem Statement/Dataset ------------------
with st.expander(r'**Problem Statement and Dataset**', expanded=False):
    st.write('''The problem statement for this dataset, from 
        [**Kaggle**](https://www.kaggle.com/datasets/aswathrao/demand-forecasting?):''')

    st.write(''' '*One of the largest retail chains in the world wants to use their vast data source to build 
        an efficient forecasting model to predict the sales for each SKU in its portfolio at its 76 different 
        stores using historical sales data for the past 3 years on a week-on-week basis. Sales and promotional 
        information is also available for each week - product and store wise. However, no other information 
        regarding stores and products are available. Can you still forecast accurately the sales values for 
        every such product/SKU-store combination for the next 12 weeks accurately?*' ''')

    st.write('These are the original columns provided in the dataset:')
    schema = [
        {'Feature':'record_ID',       'Type':'int',    'Description':'Row identifier'},
        {'Feature':'week',            'Type':'date',   'Description':'Week timestamp (YYYY-MM-DD)'},
        {'Feature':'store_id',        'Type':'int',    'Description':'Store ID number'},
        {'Feature':'sku_id',          'Type':'int',    'Description':'Product ID number'},
        {'Feature':'total_price',     'Type':'float',  'Description':'Discounted price'},
        {'Feature':'base_price',      'Type':'float',  'Description':'Regular price'},
        {'Feature':'is_featured_sku', 'Type':'0/1',    'Description':'Featured in ad that week?'},
        {'Feature':'is_display_sku',  'Type':'0/1',    'Description':'On display that week?'},
        {'Feature':'units_sold',      'Type':'int',    'Description':'Units sold that week (demand)'},
    ]
    st.table(pd.DataFrame(schema))

    st.write('''Note that the week timestamps are provided irregularly throughout the dataset, so that different 
        weeks are identified by different days of the week.''')

# -------------- Feature Engineering --------------
with st.expander(r'**Feature Engineering**', expanded=False):
    st.write('''In order to improve inference I feed the model some suggestive derived features:''')
    st.markdown(r'''
        - discount      = base_price - total_price
        - disount_pct   = discount/base_price
        - sin_dow       = sin(2$\pi$ x DOW/7)
        - cos_dow       = cos(2$\pi$ x DOW/7)
        - sin_doy       = sin(2$\pi$ x DOY/365)
        - cos_doy       = cos(2$\pi$ x DOY/365)
        ''')
    st.write(r'''where DOW is the day of the week indexed (Monday is 0), and DOY is the day of the year (0-365). 
        I expect the discount information might be a particularly predictive combination of the base and sale prices, 
        and the sines and cosines are to provide predictive information about customers' seasonal demand trends in a 
        form that is periodic, so that the model may easily associate Mon. $\approx$ Sun., and Dec. 31 $\approx$ Jan. 1.''')

# -------------- NN Architecture ----------------
with st.expander(r'**Neural Network Architecture and Validation**', expanded=False):
    st.write(r'''In order to learn features about the stores and products, I use a learnable embedding for the store_id and 
        another for the sku_id, each being mapped to a 16-dimensional vector. I then concatenate these with the 10 other features 
        to form a 42-dimensional input which is fed into a fully connected MLP ($42 \rightarrow 256 \rightarrow 256 \rightarrow 256 \rightarrow 1$) 
        with ReLU activation and 10\% dropout. The output of the last layer is log(1 + units_sold) to dampen the effects of large outliers.''')

    dot = r'''digraph G {
      rankdir=LR; nodesep=0.5; ranksep=0.5;
      node [fontname='Helvetica', color='#444', fixedsize=true];

      // ------- ID inputs and embeddings -------
      store_in [shape=box, style=filled, width=1.0, height=0.5, fillcolor='#f6f8fa', label='store_id (1)'];
      sku_in   [shape=box, style=filled, width=1.0, height=0.5, fillcolor='#f6f8fa', label='sku_id (1)'];

      emb_store [shape=box, style=filled, width=1.7, height=0.9, fillcolor='#ffe5cc',
                 label='Embedding\nstores 76×16'];
      emb_sku   [shape=box, style=filled, width=1.7, height=0.9, fillcolor='#ffe5cc',
                 label='Embedding\nskus 28×16'];

      // ------- Xcont -------
      tgt [shape=box, style=filled, fixedsize=false, fillcolor='#f6f8fa',
           label='base_price, total_price, discount,\n\ discount_pct, is_featured_sku,\n\ is_display_sku, sin_dow, cos_dow,\n\ sin_doy, cos_doy (10)'];


      // Concat node (embeddings + features)
      concat [shape=circle, style=filled, width=0.38, fillcolor='#eeeeee', label='⊕'];

      // ------- MLP -------
      h1   [shape=box, style=filled, width=0.9, height=1.3, fillcolor='#cce5ff', label='Dense 1 (256)'];
      act1 [shape=ellipse, width=0.55, height=0.35, label='ReLU'];
      drop1[shape=ellipse, width=0.8, height=0.35, label='Dropout 0.1'];

      h2   [shape=box, style=filled, width=0.9, height=1.3, fillcolor='#cce5ff', label='Dense 2 (256)'];
      act2 [shape=ellipse, width=0.55, height=0.35, label='ReLU'];
      drop2[shape=ellipse, width=0.8, height=0.35, label='Dropout 0.1'];

      h3   [shape=box, style=filled, width=0.9, height=1.3, fillcolor='#cce5ff', label='Dense 3 (256)'];
      act3 [shape=ellipse, width=0.55, height=0.35, label='ReLU'];
      drop3[shape=ellipse, width=0.8, height=0.35, label='Dropout 0.1'];

      out  [shape=box, style=filled, width=1.3, height=1.0, fillcolor='#d5f5e3',
            label='Output\nlog1p(units_sold) (1)'];

      // ------- Edges -------
      store_in -> emb_store -> concat;
      sku_in   -> emb_sku   -> concat;
      tgt      -> concat;

      concat -> h1 -> act1 -> drop1 -> h2 -> act2 -> drop2 -> h3 -> act3 -> drop3 -> out;
    }'''

    st.graphviz_chart(dot, use_container_width=True)

    st.write(r'''I used the last 8 weeks of data to form a validation set, which was comprised of 9240 samples from the original 150150. 
        This validation set was used for early stopping during training. The NN validation errors were then compared to those of LightGBM as 
        a benchmark. The NN obtained a wMAPE of 25.21\%, essentially tying LightGBM's 25.45\%.''')



# ---------- Interactive Tool -----------------
st.write(r'**Forecasting Tool**')

n_stores = len(store_ids)  # 76
n_skus   = len(sku_ids)    # 28

# Read-in product number and store number
st.write(r'Select which product at which store location you wish to forecast demand for:')
col1, col2 = st.columns(2)

with col1:
    st.markdown(f'**Product Number (1–{n_skus})**')
    sku_ord = st.slider('sku_id', 1, 28, 1, 1, key='sku_id', label_visibility='collapsed')

with col2:
    st.markdown(f'**Store Number (1–{n_stores})**')
    store_ord = st.slider('store_id', 1, 76, 1, 1, key='store_id', label_visibility='collapsed')

store_idx = store_ord - 1
sku_idx = sku_ord -1
base_lo, base_md, base_hi, disc_lo, disc_md, disc_hi, total_lo, total_hi = product_stats[sku_idx]
store_id = store_ids[store_idx]
sku_id = sku_ids[sku_idx]


# Read-in base price and discount percentage
st.write(f'''Select the base price and discount percentage you wish to set for Product {sku_id} 
        at Store {store_id}:''')
col1, col2 = st.columns(2)

with col1:
    st.markdown(f'**Base Price**')
    base_price = st.slider('base_price', base_lo, base_hi, base_md, 1.0, key='base_price', label_visibility='collapsed')

with col2:
    st.markdown(f'**Discount Percentage \% (negative = markup)**')
    discount_pct_100 = st.slider('discount_pct', 100*disc_lo, 100*disc_hi, 100*disc_md, 1.0, key='discount_pct', label_visibility='collapsed')

discount_pct = discount_pct_100/100.0
discount = discount_pct*base_price
total_price = (1.0-discount_pct)*base_price

if not (total_lo <= total_price <= total_hi):
    st.caption('**:red[Warning: This product has never been sold in this price range before, now extrapolating.]**')

# Read-in whether featured and whether on display
st.write(f'''Select whether the product was featured in ads this week, and whether it was on display:''')
col1, col2 = st.columns(2)
with col1:
    featured_toggle = st.toggle('**Featured this week?**', value=False, key='feat')
with col2:
    display_toggle  = st.toggle('**On display?**', value=False, key='disp')

is_featured = int(featured_toggle)  # 1/0 for your model
is_display  = int(display_toggle)

# Make sure order matches feat_cols
Xcont_unscaled = np.zeros((12,10), dtype=np.float32)
Xcont_unscaled[:,:6] = np.array([base_price, total_price, discount, discount_pct, is_featured, is_display])

# Get date in NY
oos.environ['TZ'] = 'America/New_York'
if hasattr(time, 'tzset'):
    time.tzset()

t   = time.localtime()
dow = t.tm_wday  
doy = t.tm_yday   

# Start predicting from next Monday
days_to_next_monday = (7 - dow) % 7
doy += days_to_next_monday
dow = 0
for i in range(12):
    sin_doy, cos_doy = np.sin(2*np.pi*doy/365.0), np.cos(2*np.pi*doy/365.0)
    sin_dow, cos_dow = np.sin(2*np.pi*dow/7.0), np.cos(2*np.pi*dow/7.0)
    Xcont_unscaled[i,6:] = np.array([sin_doy, cos_doy, sin_dow, cos_dow])
    doy += 7

store_idx_in = np.array([store_idx]*12)
sku_idx_in = np.array([sku_idx]*12)
Xcont = (Xcont_unscaled - scaler_mean)/scaler_scale

# ---------- Inference ----------
start = time.perf_counter()
nn_pred = nn_predict_cached(store_idx_in, sku_idx_in, Xcont)
elapsed_ms = (time.perf_counter() - start) * 1e3
st.info(f'NN inference time for all 12 weeks: **{elapsed_ms:.1f} ms** on CPU')

start = time.perf_counter()
lgbm_pred = lgbm_predict_cached(store_idx_in, sku_idx_in, Xcont)
elapsed_ms = (time.perf_counter() - start) * 1e3
st.info(f'LightGBM inference time for all 12 weeks: **{elapsed_ms:.1f} ms** on CPU')

weeks = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(5, 2.375))
ax.plot(weeks, nn_pred, marker='o', label='NN')
ax.plot(weeks, lgbm_pred, marker='o', label='LightGBM')
ax.set_ylim(0, 1.1*max(np.max(nn_pred), np.max(lgbm_pred)))
ax.set_xlabel('Weeks from today')
ax.set_ylabel('Units sold')
ax.set_title('12 Week Demand Forecast')
ax.legend()

st.pyplot(fig)


# ------------- Acknowledgements ------------
with st.expander('Acknowledgements'):
    st.write('''This app was created using the Demand Forecasting dataset available at [**Kaggle**](https://www.kaggle.com/datasets/aswathrao/demand-forecasting?).

        (App created by Jonathan Gordon.)''')
