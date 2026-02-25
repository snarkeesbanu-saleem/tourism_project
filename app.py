import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

DATA_FOLDER = r"C:\tourism_project\data"
MODELS_FOLDER = "models"

os.makedirs(MODELS_FOLDER, exist_ok=True)

# ────────────────────────────────────────────────
# LOAD MASTER DATA (with fallback)
# ────────────────────────────────────────────────

@st.cache_data
def load_master():
    try:
        return pd.read_parquet("master_for_app.parquet")
    except:
        st.info("Loading and merging raw data (this may take a moment)...")
        df_trans = pd.read_excel(f"{DATA_FOLDER}/Transaction.xlsx")
        df_user  = pd.read_excel(f"{DATA_FOLDER}/User.xlsx")
        df_item  = pd.read_excel(f"{DATA_FOLDER}/Item.xlsx")
        df_type  = pd.read_excel(f"{DATA_FOLDER}/Type.xlsx")
        df_mode  = pd.read_excel(f"{DATA_FOLDER}/Mode.xlsx")
        df_city  = pd.read_excel(f"{DATA_FOLDER}/City.xlsx")

        master = df_trans.merge(df_user, on='UserId', how='left') \
                         .merge(df_item, on='AttractionId', how='left') \
                         .merge(df_type, on='AttractionTypeId', how='left') \
                         .merge(df_mode, left_on='VisitMode', right_on='VisitModeId', how='left') \
                         .merge(df_city, left_on='AttractionCityId', right_on='CityId', how='left', suffixes=('', '_city'))

        master = master.rename(columns={'VisitMode_y': 'VisitMode_Name'})
        
        if 'VisitYear' in master.columns and 'VisitMonth' in master.columns:
            master['VisitYearMonth'] = master['VisitYear'].astype(str) + '-' + master['VisitMonth'].astype(str).str.zfill(2)
        
        master.to_parquet("master_for_app.parquet")
        return master

master = load_master()

# Precompute rating matrix for recommendations
rating_matrix = master.pivot_table(
    index='UserId',
    columns='AttractionId',
    values='Rating',
    fill_value=0
)

# ────────────────────────────────────────────────
# LOAD MODELS (with error handling)
# ────────────────────────────────────────────────

rf_rating = None
rf_visitmode = None
le_visitmode = None
item_similarity = None

try:
    rf_rating = joblib.load(f"{MODELS_FOLDER}/rf_rating_predictor.pkl")
except:
    st.warning("Rating prediction model not found.")

try:
    rf_visitmode = joblib.load(f"{MODELS_FOLDER}/rf_visitmode_classifier.pkl")
    le_visitmode = joblib.load(f"{MODELS_FOLDER}/visitmode_label_encoder.pkl")
except:
    st.warning("Visit mode prediction model not found.")

try:
    item_similarity = pd.read_pickle(f"{MODELS_FOLDER}/item_similarity.pkl")
except:
    st.warning("Recommendation similarity matrix not found.")

# ────────────────────────────────────────────────
# RECOMMENDATION FUNCTION
# ────────────────────────────────────────────────

def simple_recommend_for_user(user_id, n=5):
    if user_id not in rating_matrix.index or item_similarity is None:
        return []
    
    user_ratings = rating_matrix.loc[user_id]
    similar_scores = item_similarity.dot(user_ratings)
    already_rated = user_ratings[user_ratings > 0].index
    similar_scores = similar_scores[~similar_scores.index.isin(already_rated)]
    top_n = similar_scores.sort_values(ascending=False).head(n)
    
    results = []
    for attr_id, score in top_n.items():
        name_row = master[master['AttractionId'] == attr_id]['Attraction']
        name = name_row.values[0] if not name_row.empty else f"Attraction {attr_id}"
        results.append((name, score))
    return results

# ────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

st.title("Tourism Experience Analytics App")
st.markdown("Predict ratings, predict visit mode, get personalized recommendations")

# Sidebar
st.sidebar.header("User & Attraction Selection")
user_id = st.sidebar.number_input("User ID", min_value=1, max_value=1000000, value=70456, step=1)
attraction_options = sorted(master['Attraction'].dropna().unique())
selected_attraction = st.sidebar.selectbox("Select Attraction to Predict Rating", attraction_options)

# Create tabs FIRST
tab1, tab2, tab3 = st.tabs(["Predict Rating", "Predict Visit Mode", "Top Recommendations"])

# Tab 1: Predict Rating
with tab1:
    st.subheader("Predict Rating for Selected Attraction")
    
    if st.button("Predict Rating"):
        if rf_rating is None:
            st.error("Rating prediction model not loaded.")
        else:
            # Use ONLY the features the model was trained on
            # CHANGE THIS LIST to match your original training features
            training_features = ['VisitMonth', 'VisitYear', 'AttractionTypeId', 'ContinentId', 'CountryId']
            
            attr_rows = master[master['Attraction'] == selected_attraction]
            if attr_rows.empty:
                st.warning("No data for attraction – using defaults")
                row = pd.Series({col: 0 for col in training_features})
            else:
                row = attr_rows.iloc[0]
            
            input_data = pd.DataFrame([{col: row.get(col, 0) for col in training_features}])
            input_data = input_data[training_features]  # exact order & columns
            
            pred = rf_rating.predict(input_data)[0]
            st.success(f"Predicted Rating for **{selected_attraction}**: **{pred:.2f} / 5**")

# Tab 2: Predict Visit Mode
with tab2:
    st.subheader("Predict Likely Visit Mode")
    
    if st.button("Predict Visit Mode"):
        if rf_visitmode is None or le_visitmode is None:
            st.error("Visit mode model not loaded.")
        else:
            user_rows = master[master['UserId'] == user_id]
            if user_rows.empty:
                st.warning("No data for user – using sample")
                user_rows = master.sample(1)
            
            row = user_rows.iloc[0]
            input_data = pd.DataFrame([{
                'Rating': row.get('Rating', 3.5),
                'VisitMonth': row.get('VisitMonth', 6),
                'VisitYear': row.get('VisitYear', 2025),
                'AttractionTypeId': row.get('AttractionTypeId', 0),
                'ContinentId': row.get('ContinentId', 0)
            }])
            
            pred_encoded = rf_visitmode.predict(input_data)[0]
            pred_mode = le_visitmode.inverse_transform([pred_encoded])[0]
            
            st.success(f"Predicted Visit Mode for User {user_id}: **{pred_mode}**")

# Tab 3: Recommendations
with tab3:
    st.subheader(f"Top 5 Recommended Attractions for User {user_id}")
    
    if st.button("Get Recommendations"):
        recs = simple_recommend_for_user(user_id, n=5)
        
        if recs:
            for i, (name, score) in enumerate(recs, 1):
                st.write(f"**{i}.** {name} → Score: {score:.3f}")
        else:
            st.warning("No recommendations available (user has no ratings or similarity matrix missing).")

# Footer
st.markdown("---")
st.caption("Tourism Experience Analytics | Regression, Classification & Recommendation | Built with Streamlit")