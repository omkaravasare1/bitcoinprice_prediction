import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# === 1. LOAD AND CLEAN DATA ===
def load_and_clean(file):
    df = pd.read_csv(file, sep=';', thousands=',')
    cols = ['timeClose', 'open', 'high', 'low', 'close', 'volume']
    df = df[cols].rename(columns={
        'timeClose': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    df['Date'] = pd.to_datetime(df['Date'].str.replace('"', '').str[:10])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('Date').dropna().reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)
    return df

# === 2. FEATURE ENGINEERING ===
def feature_engineering(df):
    data = df.copy()
    for lag in [1, 2, 3, 5]:
        data[f'Close_Lag{lag}'] = data['Close'].shift(lag)
    for ma in [3, 5, 10]:
        data[f'MA_{ma}'] = data['Close'].rolling(ma).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility_3'] = data['Close'].rolling(3).std()
    data['Range'] = data['High'] - data['Low']
    data['Volume_MA5'] = data['Volume'].rolling(5).mean()
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna().reset_index(drop=True)
    return data

# === 3. SPLIT AND SCALE ===
def split_and_scale(data, feature_cols, test_size=0.2):
    X = data[feature_cols].values
    y = data['Target'].values
    n = len(data)
    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = RobustScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, scaler

# === 4. TRAIN MODEL ===
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, mae, rmse, r2

# === 5. NEXT DAY PREDICTION ===
def predict_next_day(model, scaler, last_row, feature_cols):
    X = scaler.transform(last_row[feature_cols].values.reshape(1, -1))
    return model.predict(X)[0]

# === 6. STREAMLIT APP ===
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.title("📈 Bitcoin Price Prediction using Linear Regression")

st.markdown("""
Upload your *CoinMarketCap Bitcoin CSV* (with columns like: timeClose, open, high, low, close, volume)  
and get insights, visualizations, and the *next-day predicted close price*.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = load_and_clean(uploaded_file)
    st.success(f"✅ Loaded {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}.")

    # Show last few rows
    with st.expander("📋 Preview Data"):
        st.dataframe(df.tail())

    # Feature Engineering
    data = feature_engineering(df)
    feature_cols = [c for c in data.columns if c not in ['Date', 'Target']]
    X_train, X_test, y_train, y_test, scaler = split_and_scale(data, feature_cols)
    model, y_pred, mae, rmse, r2 = train_linear_regression(X_train, X_test, y_train, y_test)

    # --- Metrics ---
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")

    # --- Graph 1: Prediction vs Actual ---
    st.subheader("📉 Prediction vs Actual")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(y_test, label="Actual", color="blue")
    ax1.plot(y_pred, label="Predicted", color="orange")
    ax1.set_title("BTC Price Prediction (Linear Regression)")
    ax1.legend()
    st.pyplot(fig1)

    # --- Graph 2: Residuals Plot ---
    st.subheader("🔍 Residuals Plot")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.scatter(range(len(residuals)), residuals, color="red", alpha=0.5)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_xlabel("Test Sample Index")
    ax2.set_ylabel("Residual")
    st.pyplot(fig2)

    # --- Graph 3: Histogram of Prediction Errors ---
    st.subheader("📊 Histogram of Prediction Errors")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(residuals, bins=30, color="green", alpha=0.7)
    ax3.set_xlabel("Error")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

    # --- Next Day Prediction ---
    next_day_pred = predict_next_day(model, scaler, data.iloc[[-1]], feature_cols)
    latest_close = float(data.iloc[-1]['Close'])
    change = next_day_pred - latest_close
    st.subheader("📅 Next Day Prediction")
    colA, colB, colC = st.columns(3)
    colA.metric("Latest Close", f"{latest_close:.2f} USD")
    colB.metric("Predicted Next Close", f"{next_day_pred:.2f} USD")
    colC.metric("Predicted Change", f"{change:+.2f} USD")

    st.markdown("---")
    st.caption("Model: Linear Regression | Features: Lags, Moving Averages, Volatility, Range, Volume")

else:
    st.info("👆 Please upload a Bitcoin CSV file to get started.")