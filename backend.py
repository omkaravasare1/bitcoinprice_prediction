import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === 1. LOAD AND CLEAN DATA ===
def load_and_clean(filepath):
    df = pd.read_csv(filepath, sep=';', thousands=',')
    cols = ['timeClose','open','high','low','close','volume']
    df = df[cols].rename(columns={'timeClose': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df['Date'] = pd.to_datetime(df['Date'].str.replace('"','').str[:10])
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('Date').dropna().reset_index(drop=True)
    # Remove last row
    df = df.iloc[:-1].reset_index(drop=True)
    print(f"Loaded {len(df)} rows (last row removed) from {df['Date'].min()} to {df['Date'].max()}.")
    return df

# === 2. FEATURE ENGINEERING ===
def feature_engineering(df):
    data = df.copy()
    for lag in [1,2,3,5]:
        data[f'Close_Lag{lag}'] = data['Close'].shift(lag)
    for ma in [3,5,10]:
        data[f'MA_{ma}'] = data['Close'].rolling(ma).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility_3'] = data['Close'].rolling(3).std()
    data['Range'] = data['High'] - data['Low']
    data['Volume_MA5'] = data['Volume'].rolling(5).mean()
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna().reset_index(drop=True)
    return data

# === 3. TRAIN / TEST SPLIT & SCALING ===
def split_and_scale(data, feature_cols, test_size=0.2):
    X = data[feature_cols].values
    y = data['Target'].values
    n = len(data)
    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = RobustScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, scaler

# === 4. TRAIN LINEAR REGRESSION ===
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"LinearRegression: R2={r2:.3f} ({r2*100:.2f}%), MAE={mae:.2f}, RMSE={rmse:.2f}")
    return model, y_pred, r2

# === 5. NEXT DAY PREDICTION ===
def predict_next_day(model, scaler, last_row, feature_cols):
    X = scaler.transform(last_row[feature_cols].values.reshape(1,-1))
    pred = model.predict(X)[0]
    return pred

# === 6. MAIN PIPELINE ===
def main(filepath):
    df = load_and_clean(filepath)
    data = feature_engineering(df)
    feature_cols = [c for c in data.columns if c not in ['Date','Target']]
    X_train, X_test, y_train, y_test, scaler = split_and_scale(data, feature_cols)
    model, y_pred, r2 = train_linear_regression(X_train, X_test, y_train, y_test)
    
    # --- Graph 1: Prediction vs Actual ---
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title('BTC Price Prediction (Linear Regression)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('btc_pred_plot.png')
    
    # --- Graph 2: Residuals Plot ---
    residuals = y_test - y_pred
    plt.figure(figsize=(10,5))
    plt.scatter(range(len(residuals)), residuals, color='red', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals of Predictions')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig('btc_residuals_plot.png')
    
    # --- Graph 3: Histogram of Prediction Errors ---
    plt.figure(figsize=(8,4))
    plt.hist(residuals, bins=30, color='green', alpha=0.7)
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('btc_error_hist.png')
    
    print("Plots saved: 'btc_pred_plot.png', 'btc_residuals_plot.png', 'btc_error_hist.png'.")
    
    # Next day prediction using last row of data before removal
    next_day = predict_next_day(model, scaler, data.iloc[[-1]], feature_cols)
    print(f"\nLatest Close: {data.iloc[-1]['Close']:.2f} | Next-day Predicted: {next_day:.2f} | Change: {next_day-data.iloc[-1]['Close']:.2f}")

if __name__ == "-__main__":
    main("Bitcoin_10_10_2024-10_10_2025_historical_data_coinmarketcap.csv")