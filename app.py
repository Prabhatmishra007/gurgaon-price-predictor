import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Gurgaon Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

    .main { background-color: #0f0f0f; }

    h1 { color: #e8c97a !important; font-size: 2.4rem !important; font-weight: 300 !important; }
    h3 { color: #c9a84c !important; font-size: 1rem !important;
         letter-spacing: 0.15em; text-transform: uppercase; margin-top: 1.5rem !important; }

    .stSelectbox label, .stNumberInput label, .stRadio label  {
        font-size: 13px !important; color: #9a9490 !important;
        text-transform: uppercase; letter-spacing: 0.08em;
    }

    .result-box {
        background: #1a1a1a;
        border: 1px solid rgba(201,168,76,0.4);
        border-radius: 8px;
        padding: 32px 36px;
        margin-top: 24px;
        text-align: center;
    }
    .result-label { font-size: 12px; color: #7a6230; letter-spacing: 0.2em;
                    text-transform: uppercase; margin-bottom: 8px; }
    .result-price { font-size: 3.5rem; color: #e8c97a; font-weight: 300;
                    line-height: 1; margin: 8px 0; }
    .result-unit  { font-size: 1.1rem; color: #c9a84c; }
    .result-range { font-size: 14px; color: #7a7670; margin-top: 10px; }
    .result-range span { color: #b0a898; }
    .meta-row { display: flex; justify-content: center; gap: 40px;
                margin-top: 20px; padding-top: 16px;
                border-top: 1px solid #2a2a2a; flex-wrap: wrap; }
    .meta-item { text-align: center; }
    .meta-key { font-size: 11px; color: #7a7670; text-transform: uppercase;
                letter-spacing: 0.1em; }
    .meta-val { font-size: 15px; color: #e8e2d6; margin-top: 3px; }

    div[data-testid="stHorizontalBlock"] { gap: 16px; }
    .stButton > button {
        width: 100%; background: transparent;
        border: 1px solid #c9a84c; color: #c9a84c;
        font-family: 'Nunito', sans-serif; font-size: 14px;
        letter-spacing: 0.15em; text-transform: uppercase;
        padding: 14px; border-radius: 4px;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: #c9a84c; color: #0a0a0a; }
</style>
""", unsafe_allow_html=True)


# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
    return pipeline, df

pipeline, df = load_model()

# ── Unique values from df ─────────────────────────────────────
sectors         = sorted(df['sector'].unique().tolist())
age_options     = sorted(df['agePossession'].unique().tolist())
balcony_options = sorted(df['balcony'].unique().tolist())
furnish_options = ['unfurnished', 'semifurnished', 'furnished']
luxury_options  = ['Low', 'Medium', 'High']
floor_options   = ['Low Floor', 'Mid Floor', 'High Floor']
bed_options     = sorted(df['bedRoom'].unique().tolist())
bath_options    = sorted(df['bathroom'].unique().tolist())


# ── Header ───────────────────────────────────────────────────
st.markdown("<h1>🏙️ Gurgaon Property<br>Price Predictor</h1>", unsafe_allow_html=True)
st.caption("Powered by a Random Forest model trained on Gurgaon real estate data")
st.divider()


# ── Form ─────────────────────────────────────────────────────
st.markdown("<h3>Property Details</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    property_type = st.selectbox("Property Type", ["flat", "house"],
                                  format_func=str.capitalize)
with col2:
    sector = st.selectbox("Sector / Location", sectors,
                           format_func=str.title)

col3, col4, col5 = st.columns(3)
with col3:
    bedRoom = st.selectbox("Bedrooms", bed_options,
                            index=bed_options.index(3.0) if 3.0 in bed_options else 0,
                            format_func=lambda x: f"{int(x)} BHK")
with col4:
    bathroom = st.selectbox("Bathrooms", bath_options,
                             index=bath_options.index(3.0) if 3.0 in bath_options else 0,
                             format_func=lambda x: f"{int(x)}")
with col5:
    balcony = st.selectbox("Balconies", balcony_options)

col6, col7 = st.columns(2)
with col6:
    built_up_area = st.number_input("Built-up Area (sq ft)", min_value=33,
                                     max_value=15000, value=1500, step=50)
with col7:
    agePossession = st.selectbox("Age / Possession", age_options)

st.markdown("<h3>Amenities & Classification</h3>", unsafe_allow_html=True)

col8, col9, col10 = st.columns(3)
with col8:
    furnishing_type = st.selectbox("Furnishing", furnish_options,
                                    format_func=str.capitalize)
with col9:
    luxury_category = st.selectbox("Luxury Category", luxury_options)
with col10:
    floor_category = st.selectbox("Floor", floor_options)

col11, col12 = st.columns(2)
with col11:
    servant_room = st.radio("Servant Room", [0, 1],
                             format_func=lambda x: "Yes" if x else "No",
                             horizontal=True)
with col12:
    store_room = st.radio("Store Room", [0, 1],
                           format_func=lambda x: "Yes" if x else "No",
                           horizontal=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────
if st.button("✦  Estimate Price"):
    input_df = pd.DataFrame([[
        property_type, sector, float(bedRoom), float(bathroom),
        balcony, agePossession, float(built_up_area),
        float(servant_room), float(store_room),
        furnishing_type, luxury_category, floor_category
    ]], columns=df.columns)

    log_price = pipeline.predict(input_df)[0]
    price_cr  = np.expm1(log_price)          # in Crores
    price_lakh = price_cr * 100              # in Lakhs
    low_cr  = price_cr * 0.88
    high_cr = price_cr * 1.12
    ppsf = (price_lakh * 100000) / built_up_area

    def fmt(val_cr):
        if val_cr >= 1:
            return f"₹ {val_cr:.2f} Cr"
        return f"₹ {val_cr*100:.1f} L"

    main_str = fmt(price_cr)
    range_str = f"{fmt(low_cr)} – {fmt(high_cr)}"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Market Value</div>
        <div class="result-price">{main_str}</div>
        <div class="result-range">Estimated range &nbsp;<span>{range_str}</span></div>
        <div class="meta-row">
            <div class="meta-item">
                <div class="meta-key">Price / sq ft</div>
                <div class="meta-val">₹ {int(ppsf):,}</div>
            </div>
            <div class="meta-item">
                <div class="meta-key">Built-up Area</div>
                <div class="meta-val">{int(built_up_area):,} sq ft</div>
            </div>
            <div class="meta-item">
                <div class="meta-key">Configuration</div>
                <div class="meta-val">{int(bedRoom)} BHK · {int(bathroom)} Bath</div>
            </div>
            <div class="meta-item">
                <div class="meta-key">Location</div>
                <div class="meta-val">{sector.title()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Predictions are estimates based on historical Gurgaon property data · For reference only")
