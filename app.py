import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Eco-Urban Predictor", layout="wide", initial_sidebar_state="collapsed")

# 2. CSS - Dark & Lime Glow
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #020a02 0%, #000000 100%);
        background-attachment: fixed;
    }

    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    /* Glass card styling */
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(191, 255, 0, 0.3);
        margin-bottom: 15px;
    }

    [data-testid="stMetricValue"] {
        color: #bfff00 !important; 
        font-size: 1.8rem !important;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(191, 255, 0, 0.5);
    }
    
    h1 { color: white !important; font-size: 1.8rem !important; }
    h3 { color: #bfff00 !important; font-size: 1.2rem !important; margin-bottom: 10px !important; }
    label, p, .stWrite { color: #cccccc !important; }

    #MainMenu, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Load Assets
@st.cache_resource
def load_assets():
    try:
        reg_model = joblib.load('models/environmental_regressor.pkl')
        cls_model = joblib.load('models/urbanization_classifier.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        le = joblib.load('models/district_encoder.pkl')
        return reg_model, cls_model, scaler, le
    except:
        return None, None, None, None

reg_model, cls_model, scaler, le = load_assets()

if reg_model:
    st.markdown("<h1>🌿 Eco-Urban Intelligence Dashboard</h1>", unsafe_allow_html=True)
    
    # --- Top Card ---
    with st.container():
        st.markdown("<h3>🌍 Area & Timeline Selection</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            selected_district = st.selectbox("Select District", le.classes_)
        with c2:
            target_year = st.select_slider("Prediction Year", options=list(range(2025, 2051)), value=2030)

    # Logic
    dist_idx = le.transform([selected_district])[0]
    reg_input = pd.DataFrame([[dist_idx, target_year]], columns=['District_Encoded', 'Year'])
    factors = reg_model.predict(reg_input)[0] 
    y_diff = target_year - 2024
    p_lst = factors[0] + (y_diff * 0.12)
    p_ndvi = max(0.1, factors[1] - (y_diff * 0.004))
    cls_in = scaler.transform([[p_lst, p_ndvi, factors[3]]])
    prob = cls_model.predict_proba(cls_in)[0][1] * 100

    # Content
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("<h3>📊 Environmental Metrics</h3>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("LST (Temp)", f"{p_lst:.2f} °C")
            m2.metric("NDVI (Green)", f"{p_ndvi:.4f}")
            
            # Pie Chart with Percentage in Middle
            fig = px.pie(values=[prob, 100-prob], 
                         names=['Risk', 'Stable'], 
                         hole=0.7, height=220, 
                         color_discrete_sequence=['#FF0000', '#bfff00']) # Red for Risk
            
            fig.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), 
                paper_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text=f'{int(prob)}%', x=0.5, y=0.5, font_size=26, showarrow=False, font_color="white")]
            )
            fig.update_traces(textinfo='none')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        with st.container():
            st.markdown("<h3>⚠️ Risk Assessment</h3>", unsafe_allow_html=True)
            st.metric("Urbanization Probability", f"{prob:.2f}%")
            
            if prob > 70:
                st.error("🚨 High Urban Risk Forecasted")
            elif prob > 35:
                st.warning("🟡 Moderate Urban Expansion")
            else:
                st.success("✅ Environmentally Stable Area")
            
          
            
            # Note inside the right-bottom card

            st.markdown(f"""
                <div style="background: rgba(0,0,0,0.3); 
                            padding: 10px 10px 10px 10px; /* Top, Right, Bottom, Left */
                            border-radius: 8px; 
                            border-left: 4px solid #bfff00; 
                            margin-top: 15px;">
                    <p style="font-size: 0.8rem; margin: 0; color: #bbb !important;">
                        ⚠️ <b>Disclaimer:</b> Predicted by ML model. Data sources include historical environmental records. Actual future values may vary based on policy changes.
                    </p>
                   
                </div> <br>
            """, unsafe_allow_html=True)

else:
    st.error("Model files not found!")