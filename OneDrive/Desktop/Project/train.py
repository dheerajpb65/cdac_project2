import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Model Training Pipeline...")

# Configuration
TICKERS = ['AAPL', 'MSFT', 'TSLA', 'RELIANCE.NS', 'NVDA']
LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32

print(f"\n📥 Loading data...")

all_data = {}

# Load Apple from CSV - find where data starts
try:
    df_apple = pd.read_csv('apple.csv')
    # Find the row where actual data starts
    data_start = 0
    for i, row in df_apple.iterrows():
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).startswith('20'):
            data_start = i
            break
    
    df_apple = df_apple.iloc[data_start:].reset_index(drop=True)
    df_apple.columns = ['Date'] + list(df_apple.columns[1:])
    df_apple['Date'] = pd.to_datetime(df_apple['Date'], errors='coerce')
    df_apple = df_apple.dropna(subset=['Date'])
    df_apple = df_apple.set_index('Date')
    df_apple['Close'] = pd.to_numeric(df_apple['Close'], errors='coerce')
    all_data['AAPL'] = df_apple[['Close']].dropna()
    print(f"   ✓ AAPL (from apple.csv): {len(all_data['AAPL'])} records")
except Exception as e:
    print(f"   ✗ AAPL (apple.csv): {str(e)}")

# Download data for other stocks
for ticker in TICKERS[1:]:
    try:
        df = yf.download(ticker, start='2010-01-01', end='2024-01-01', progress=False)
        if not df.empty:
            all_data[ticker] = df[['Close']].dropna()
            print(f"   ✓ {ticker}: {len(all_data[ticker])} records")
    except Exception as e:
        print(f"   ⚠ {ticker}: Skipped")

if not all_data:
    print("\n❌ ERROR: No data loaded!")
    exit(1)

# Combine all data for training
print("\n🔄 Preparing training data...")
combined_data = pd.concat([all_data[ticker] for ticker in all_data.keys()])
close_prices = combined_data[['Close']].values

# Scale the data
main_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = main_scaler.fit_transform(close_prices)

# Create training sequences
X_train, y_train = [], []
for i in range(LOOKBACK, len(scaled_data)):
    X_train.append(scaled_data[i-LOOKBACK:i, 0])
    y_train.append(scaled_data[i, 0])

X_train = np.reshape(np.array(X_train), (len(X_train), LOOKBACK, 1))
y_train = np.array(y_train)

print(f"   ✓ Training sequences created: {len(X_train)} samples")

# Build LSTM model
print("\n🧠 Building LSTM model...")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("   ✓ Model compiled")

# Train the model
print("\n⏳ Training model (this may take a few minutes)...")
early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=0)
print("   ✓ Model training completed")

# Save the model
model.save('global_stock_model.keras')
print("\n💾 Model saved: global_stock_model.keras")

# Create and save scalers for each stock
print("\n🔧 Creating stock-specific scalers...")
scalers_dict = {}
for ticker in TICKERS:
    if ticker in all_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(all_data[ticker].values)
        scalers_dict[ticker] = scaler

# Save scalers
with open('scalers_dict.pkl', 'wb') as f:
    pickle.dump(scalers_dict, f)
print(f"   ✓ Scalers created for {len(scalers_dict)} stocks")
print("💾 Scalers saved: scalers_dict.pkl")

print("\n✅ Training complete! You can now run: streamlit run model.py")
