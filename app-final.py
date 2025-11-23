import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config & Styling ----------------
st.set_page_config(
    page_title="RainCast USA",
    page_icon="🌧️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    .main { padding: 2rem; }

    /* Sets Entire App Background */
    .stApp {
        background: linear-gradient(to bottom, #2C558C, #0f1b3d);
    }

    div[data-baseweb="select"] > div {
        color: #111827 !important;
    }
            
    ul[role="listbox"] li {
        color: #111827 !important;
    }

    div[data-baseweb="select"] span {
        color: #111827 !important;
    }
    
    /* Keep titles colorful */
    .header-title {
        font-size: 3.5rem !important;
        font-weight: 700;
        background: linear-gradient(to top, #0766DA, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    h3, [data-testid="stVerticalBlock"] h3, .css-1y0t0as h3 {
        color: #f1f5f9 !important;   /* Bright white */
        font-weight: 600 !important;
    }
            
    /* Keep input values dark so defaults are readable */
    [data-baseweb="input"] input {
        color: #111827 !important;  /* Dark text for number inputs and text fields */
    }
            
    /* Slightly dim subtitles only */
    .header-subtitle { color: #cbd5e1; }

    /* Input card: glass blur, rounded corners, semi-transparent overlay */
    .input-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Make Streamlit labels clearly visible */
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load Artifacts ----------------


@st.cache_resource
def load_artifacts():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open("threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    return model, scaler, feature_columns, threshold


model, scaler, feature_columns, threshold = load_artifacts()

# ---------------- Header ----------------
st.markdown("<h1 class='header-title'>🌧️ RainCast USA</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Will it rain tomorrow in your city? Get accurate predictions using today's weather!</p>", unsafe_allow_html=True)

# City options
city_options = {
    "Austin": "austin",
    "Charlotte": "charlotte",
    "Chicago": "chicago",
    "Columbus": "columbus",
    "Dallas": "dallas",
    "Denver": "denver",
    "Fort Worth": "fort worth",
    "Houston": "houston",
    "Indianapolis": "indianapolis",
    "Jacksonville": "jacksonville",
    "Los Angeles": "los angeles",
    "New York": "new york",
    "Philadelphia": "philadelphia",
    "Phoenix": "phoenix",
    "San Antonio": "san antonio",
    "San Diego": "san diego",
    "San Francisco": "san francisco",
    "San Jose": "san jose",
    "Seattle": "seattle",
    "Washington D.C.": "washington d.c."
}

# ---------------- Input Form in Card ----------------
with st.container():
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.subheader("📍 Select City & Today's Weather")

    location = st.selectbox("City", options=list(
        city_options.keys()), index=6)  # Default: Fort Worth

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input(
            "🌡️ Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5)
        humidity = st.number_input(
            "💧 Humidity (%)", min_value=0, max_value=100, value=65, step=1)
        wind_speed = st.number_input(
            "💨 Wind Speed (km/h)", min_value=0.0, value=12.0, step=0.5)

    with col2:
        precipitation = st.number_input(
            "🌧️ Precipitation Today (mm)", min_value=0.0, value=0.0, step=0.1)
        cloud_cover = st.number_input(
            "☁️ Cloud Cover (%)", min_value=0, max_value=100, value=45, step=5)
        pressure = st.number_input(
            "🌀 Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.5)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction Function ----------------


def predict_rain_tomorrow(input_dict):
    df = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    df["temperature"] = input_dict["temperature"]
    df["humidity"] = input_dict["humidity"]
    df["wind_speed"] = input_dict["wind_speed"]
    df["precipitation"] = input_dict["precipitation"]
    df["cloud_cover"] = input_dict["cloud_cover"]
    df["pressure"] = input_dict["pressure"]

    location_key = f"location_{input_dict['location']}"
    if location_key not in df.columns:
        return None, None, f"City '{input_dict['location']}' not supported by model."
    df[location_key] = 1

    numeric_cols = ['temperature', 'humidity', 'wind_speed',
                    'precipitation', 'cloud_cover', 'pressure']
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    prob = model.predict_proba(df_scaled)[0, 1]
    # pred = int(prob >= threshold)

    return prob, None


# ---------------- Predict Button ----------------
if st.button("🔮 Predict Tomorrow's Weather", use_container_width=True, type="primary"):
    with st.spinner("Analyzing weather patterns..."):
        selected_city_key = city_options[location]

        input_dict = {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
            "cloud_cover": cloud_cover,
            "pressure": pressure,
            "location": selected_city_key
        }

        prob, error = predict_rain_tomorrow(input_dict)

        if error:
            st.error(error)
        else:
            # Dynamic result styling
            if prob >= 0.6:
                icon = "☔"
                title = "Rain Expected Tomorrow!"
                color = "#CE1212"
                light_bg = "#F58E8E"
                desc = "Make sure you have an umbrella!"
                risk = "High"
            elif prob >= 0.3:
                icon = "☔"
                title = "It Might Rain Tomorrow!"
                color = "#EE6F2F"
                light_bg = "#FFAB6B"
                desc = "Bring an umbrella just in case!"
                risk = "Medium"
            else:
                icon = "🌞"
                title = "Clear Skies Tomorrow!"
                color = "#0B835B"
                light_bg = "#86D5B5"
                desc = "Perfect day for outdoor plans!"
                risk = "Low"

            risk_color = {"Low": "#0B835B",
                          "Medium": "#EE6F2F",
                          "High": "#CE1212"}[risk]

            st.markdown(f"""
            <div style="
                background: {light_bg};
                border: 3px solid {color};
                border-radius: 20px;
                padding: 2rem;
                text-align: center;
                max-width: 600px;
                margin: 2rem auto;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(8px);
            ">
                <h2 style="
                    color: {color};
                    margin: 0 0 0.8rem 0;
                    font-size: 2.3rem;
                    font-weight: 700;
                ">
                    {icon} {title}
                </h2>
                <p style="
                    font-size: 1.3rem;
                    color: #1e293b;
                    margin: 0;
                    font-weight: 500;
                ">
                    {desc}
                </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div>
                    <h3 style='text-align:center; margin:0; padding: 0;'>Probability of Rain</h4>
                    <h1 style='text-align:center; color:white; font-size:3.5rem; padding: 0;'>
                        {prob:.1%}</h1>
                    <p style='text-align:center; color:{risk_color}; font-size:1.3rem; padding: 0;'>
                        {risk} Risk</p>
                </div>
                """, unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b; font-size:0.9rem;'>"
            "Built by Kiara De Villa, Gladys San Gabriel, and Arlene Valenzuela</p>",
            unsafe_allow_html=True)
