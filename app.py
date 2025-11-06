# ========================
# IMPORTS
# ========================
import os
import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go
import math

# ========================
# CONFIGURATION
# ========================
API_KEY = "579b464db66ec23bdd000001cff99e32ded74c1e60ee264b550dd5c6"
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
TRAIN_EPOCHS = 10
FORECAST_HORIZON = 30
LOOK_BACK = 7

FORECAST_DIR = "forecasts"
os.makedirs(FORECAST_DIR, exist_ok=True)

# ========================
# PAGE SETUP
# ========================
st.set_page_config(
    page_title="🌾 Commodity Price Predictor", 
    page_icon="🌱", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main container styling */
    .main { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
        padding: 2rem; 
    }
    
    /* Header styling */
    h1 { 
        text-align: center; 
        color: #1b5e20;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #2e7d32;
        border-left: 4px solid #4caf50;
        padding-left: 1rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #388e3c;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #558b2f;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Button styling */
    .stButton>button { 
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        color: white; 
        border-radius: 12px; 
        border: none; 
        padding: 12px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover { 
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.4);
    }
    
    /* Metric box styling */
    .metric-box { 
        background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%);
        border-radius: 16px; 
        padding: 24px; 
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        border: 2px solid #c5e1a5;
        transition: transform 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0px 8px 20px rgba(0,0,0,0.12);
    }
    .metric-box h4 {
        color: #2e7d32;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 0.5rem 0;
    }
    
    /* Info boxes */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 1rem 0;
        border-left: 4px solid #4caf50;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
        border: 2px solid #c5e1a5;
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(135deg, #4caf50 0%, #81c784 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        margin: 2rem 0;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Section divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #4caf50, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&limit=10000"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["records"])
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True)
    df["avg_price"] = (
        df[["Min_Price", "Max_Price", "Modal_Price"]].astype(float).mean(axis=1)
    )
    df = df.dropna(subset=["Commodity_Code", "avg_price", "Commodity"])
    return df

df = load_data()

# ========================
# HELPERS
# ========================
def prepare_series(df, code):
    temp = df[df["Commodity_Code"] == code][["Arrival_Date", "avg_price"]]
    temp = temp.groupby("Arrival_Date").mean().reset_index()
    temp = temp.sort_values("Arrival_Date")
    temp = temp.rename(columns={"Arrival_Date": "timestamp", "avg_price": "value"})
    return temp

def fill_dates(series_df):
    all_days = pd.date_range(series_df["timestamp"].min(), series_df["timestamp"].max(), freq="D")
    series_df = series_df.set_index("timestamp").reindex(all_days).interpolate().reset_index()
    series_df.columns = ["timestamp", "value"]
    return series_df

def create_dataset(series, look_back=10):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i : i + look_back])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

# ========================
# UI HEADER
# ========================
st.title("🌾 Commodity Price Predictor")
st.markdown('<p class="subtitle">Predict future commodity prices using Deep Learning (LSTM), Facebook Prophet, and hybrid ensemble modeling</p>', unsafe_allow_html=True)

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.header("📊 Forecast Settings")
    st.markdown("---")
    
    commodity_list = df["Commodity"].unique()
    selected_commodity = st.selectbox(
        "🌾 Select Commodity:", 
        sorted(commodity_list),
        help="Choose a commodity to forecast"
    )
    
    st.markdown("---")
    st.subheader("⚙️ Model Configuration")
    st.info(f"""
    **Current Settings:**
    - Forecast Horizon: {FORECAST_HORIZON} days
    - Look Back Period: {LOOK_BACK} days
    - Training Epochs: {TRAIN_EPOCHS}
    - Hybrid Weight: 60% LSTM, 40% Prophet
    """)
    
    st.markdown("---")
    st.subheader("📈 About the Models")
    with st.expander("LSTM Neural Network"):
        st.write("Deep learning model that learns temporal patterns in price data")
    with st.expander("Prophet"):
        st.write("Facebook's time series forecasting tool optimized for business data")
    with st.expander("Hybrid Ensemble"):
        st.write("Combines both models for improved accuracy and stability")

# ========================
# MAIN CONTENT
# ========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    forecast_button = st.button("🔮 Generate Forecast", use_container_width=True)

if forecast_button:
    code = df[df["Commodity"] == selected_commodity]["Commodity_Code"].iloc[0]
    
    # Info card
    st.markdown(f'<div class="info-card">🔄 Processing <b>{selected_commodity}</b> (Code: {code})...</div>', unsafe_allow_html=True)

    # Prepare and clean data
    series = fill_dates(prepare_series(df, code))
    values = series["value"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = create_dataset(scaled, LOOK_BACK)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ========================
    # PROGRESS TRACKING
    # ========================
    progress_bar = st.progress(0)
    status_text = st.empty()

    # ========================
    # LSTM MODEL
    # ========================
    status_text.text("🤖 Training LSTM model...")
    progress_bar.progress(20)
    
    model = Sequential([
        LSTM(32, input_shape=(LOOK_BACK, 1), activation="tanh"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=TRAIN_EPOCHS, batch_size=16, verbose=0)
    
    progress_bar.progress(40)
    status_text.text("✅ LSTM training complete")

    # Forecast with LSTM
    last_seq = scaled[-LOOK_BACK:]
    preds = []
    for _ in range(FORECAST_HORIZON):
        pred = model.predict(last_seq.reshape(1, LOOK_BACK, 1), verbose=0)
        preds.append(pred[0][0])
        last_seq = np.append(last_seq[1:], pred, axis=0)
    forecast_lstm = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # ========================
    # PROPHET MODEL
    # ========================
    status_text.text("📈 Training Prophet model...")
    progress_bar.progress(60)
    
    prophet_df = series.rename(columns={"timestamp": "ds", "value": "y"})
    model_prophet = Prophet()
    model_prophet.fit(prophet_df)
    future = model_prophet.make_future_dataframe(periods=FORECAST_HORIZON)
    forecast_prophet = model_prophet.predict(future)
    prophet_values = forecast_prophet["yhat"].tail(FORECAST_HORIZON).values
    
    progress_bar.progress(80)
    status_text.text("✅ Prophet training complete")

    # ========================
    # HYBRID (FINAL) FORECAST
    # ========================
    status_text.text("🌿 Generating hybrid forecast...")
    progress_bar.progress(90)
    
    final_pred = 0.6 * forecast_lstm + 0.4 * prophet_values
    
    progress_bar.progress(100)
    status_text.text("✅ Forecast complete!")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ========================
    # VISUALIZATION
    # ========================
    st.subheader("📊 Forecast Visualization")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Hybrid Forecast", "🔍 Model Comparison", "📉 Historical Context"])
    
    with tab1:
        fig_final = go.Figure()
        fig_final.add_trace(go.Scatter(
            y=final_pred, 
            mode="lines+markers", 
            name="Hybrid Forecast", 
            line=dict(color="#2e7d32", width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(46, 125, 50, 0.1)'
        ))
        fig_final.update_layout(
            title=f"🌾 30-Day Hybrid Forecast for {selected_commodity}",
            xaxis_title="Days Ahead", 
            yaxis_title="Predicted Price (₹)",
            plot_bgcolor="white", 
            hovermode="x unified",
            height=500,
            font=dict(size=12)
        )
        st.plotly_chart(fig_final, use_container_width=True)
    
    with tab2:
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            y=forecast_lstm, 
            mode="lines", 
            name="LSTM", 
            line=dict(color="#1976d2", dash="dot")
        ))
        fig_compare.add_trace(go.Scatter(
            y=prophet_values, 
            mode="lines", 
            name="Prophet", 
            line=dict(color="#f57c00", dash="dash")
        ))
        fig_compare.add_trace(go.Scatter(
            y=final_pred, 
            mode="lines+markers", 
            name="Hybrid Final", 
            line=dict(color="#2e7d32", width=3)
        ))
        fig_compare.update_layout(
            title="Model Comparison",
            xaxis_title="Days Ahead", 
            yaxis_title="Predicted Price (₹)",
            plot_bgcolor="white", 
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    
    with tab3:
        # Historical + Forecast
        fig_hist = go.Figure()
        hist_days = min(90, len(values))
        fig_hist.add_trace(go.Scatter(
            y=values[-hist_days:].flatten(),
            mode="lines",
            name="Historical Prices",
            line=dict(color="#616161")
        ))
        fig_hist.add_trace(go.Scatter(
            y=final_pred,
            mode="lines",
            name="Forecast",
            line=dict(color="#2e7d32", width=2)
        ))
        fig_hist.update_layout(
            title=f"Historical Prices (Last {hist_days} days) + Forecast",
            xaxis_title="Time Period",
            yaxis_title="Price (₹)",
            plot_bgcolor="white",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ========================
    # METRICS
    # ========================
    st.subheader("📊 Model Performance Metrics")
    
    # Calculate metrics (comparing last 30 days if available)
    if len(values) >= FORECAST_HORIZON:
        mae_lstm = mean_absolute_error(values[-FORECAST_HORIZON:], forecast_lstm[-FORECAST_HORIZON:])
        rmse_lstm = math.sqrt(mean_squared_error(values[-FORECAST_HORIZON:], forecast_lstm[-FORECAST_HORIZON:]))
        mae_prophet = mean_absolute_error(values[-FORECAST_HORIZON:], prophet_values)
        rmse_prophet = math.sqrt(mean_squared_error(values[-FORECAST_HORIZON:], prophet_values))
    else:
        mae_lstm = rmse_lstm = mae_prophet = rmse_prophet = 0

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="metric-box">
            <h4>🤖 LSTM Model</h4>
            <div class="metric-value">MAE: ₹{mae_lstm:.2f}</div>
            <div>RMSE: ₹{rmse_lstm:.2f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-box">
            <h4>📈 Prophet Model</h4>
            <div class="metric-value">MAE: ₹{mae_prophet:.2f}</div>
            <div>RMSE: ₹{rmse_prophet:.2f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_pred = np.mean(final_pred)
        min_pred = np.min(final_pred)
        max_pred = np.max(final_pred)
        st.markdown(f'''
        <div class="metric-box">
            <h4>🌿 Hybrid Forecast</h4>
            <div class="metric-value">Avg: ₹{avg_pred:.2f}</div>
            <div>Range: ₹{min_pred:.2f} - ₹{max_pred:.2f}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ========================
    # FORECAST TABLE
    # ========================
    st.subheader("📅 Detailed Forecast Data")
    
    forecast_df = pd.DataFrame({
        "Day": np.arange(1, FORECAST_HORIZON + 1),
        "LSTM_Forecast": forecast_lstm,
        "Prophet_Forecast": prophet_values,
        "Final_Hybrid": final_pred
    })

    st.dataframe(
        forecast_df.style.format({
            "LSTM_Forecast": "₹{:.2f}",
            "Prophet_Forecast": "₹{:.2f}",
            "Final_Hybrid": "₹{:.2f}"
        }).background_gradient(subset=["Final_Hybrid"], cmap="Greens"),
        use_container_width=True,
        height=400
    )

    # ========================
    # SUMMARY & DOWNLOAD
    # ========================
    st.markdown(f'''
    <div class="success-banner">
        🎯 Average Predicted Price (Hybrid): ₹{avg_pred:.2f}
    </div>
    ''', unsafe_allow_html=True)

    filename = f"{selected_commodity.replace(' ', '_')}_{code}_forecast.csv"
    forecast_df.to_csv(os.path.join(FORECAST_DIR, filename), index=False)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            "📥 Download Forecast CSV", 
            data=forecast_df.to_csv(index=False), 
            file_name=filename,
            use_container_width=True
        )

# ========================
# FOOTER
# ========================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #558b2f; padding: 2rem;">
    <p>🌱 Powered by LSTM Neural Networks & Facebook Prophet | Data from Government of India API</p>
</div>
""", unsafe_allow_html=True)