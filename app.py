import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Equity Predictor", layout="wide")
st.header('EQUITY MARKET PREDICTION AND ANALYSIS')

# --- STEP 1: DEFINE COMPANY MAPPING ---
# Key = Display Name, Value = (Ticker for Yahoo, Model Filename)
service_companies = {
    'INFY': ('INFY.NS', 'infosys.hdf5'),
    'Wipro': ('WIT', 'wipro.hdf5'),
    'TCS': ('TCS.NS', 'tcs.hdf5'),
    'Cognizant': ('CTSH', 'cognizant.hdf5'),
    'Oracle': ('ORCL', 'oracle.hdf5'),
    'Goldman Sachs': ('GS', 'goldmansachs.hdf5'),
    'Teradata': ('TDC', 'teradata.hdf5'),
    'Capgemini': ('CAP.PA', 'capgemini.hdf5'),
    'LTI': ('LTIM.NS', 'lti.hdf5') # LTI merged into LTIM
}

product_companies = {
    'Apple': ('AAPL', 'apple.hdf5'),
    'Alibaba': ('BABA', 'baba.hdf5'),
    'Google': ('GOOGL', 'Google.hdf5'),
    'Microsoft': ('MSFT', 'microsoft.hdf5'),
    'Amazon': ('AMZN', 'amazon.hdf5'),
    'Adobe': ('ADBE', 'adobe.hdf5'),
    'Dell': ('DELL', 'dell.hdf5'),
    'HP': ('HPQ', 'hp.hdf5'),
    'Sony': ('SONY', 'sony.hdf5')
}

# --- SIDEBAR ---
select_type = st.sidebar.radio('Select Company Type', ['Service Company', 'Product Company'])

if select_type == 'Service Company':
    selected_name = st.sidebar.selectbox('Select Company', list(service_companies.keys()))
    ticker, model_file = service_companies[selected_name]
else:
    selected_name = st.sidebar.selectbox('Select Company', list(product_companies.keys()))
    ticker, model_file = product_companies[selected_name]

# --- STEP 2: MODEL LOADING ---
def load_my_model(path):
    # Manual build to avoid ndim=4 metadata error
    model = Sequential([
        Input(shape=(100, 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    try:
        model.load_weights(path)
        return model
    except Exception as e:
        st.error(f"Weights file '{path}' not found or incompatible.")
        return None

model = load_my_model(model_file)
if model is None: st.stop()

# --- STEP 3: DATA FETCHING ---

@st.cache_data
def load_data(symbol):
    # Historical data range from your original project
    data = yf.download(symbol, start='2015-05-27', end='2020-05-22')
    return data

df = load_data(ticker)

if df.empty:
    st.error(f"No data found for ticker '{ticker}'. This usually happens if the ticker symbol is incorrect for the date range.")
    st.stop()

# --- STEP 4: PREPROCESSING ---
df_close = df.reset_index()['Close']
scaler = MinMaxScaler(feature_range=(0,1))
# This is where your previous ValueError occurred - now protected by the df.empty check
df1_scaled = scaler.fit_transform(np.array(df_close).reshape(-1,1))

training_size = int(len(df1_scaled) * 0.65)
test_data = df1_scaled[training_size:]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        dataX.append(dataset[i:(i+time_step), 0])
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_test, y_test = create_dataset(test_data, time_step)

# Check if we have enough test data to predict
if len(X_test) == 0:
    st.warning("Not enough data in the test set to create a prediction window. Try a wider date range.")
    st.stop()

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --- STEP 5: PREDICTIONS & FORECAST ---
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

# Forecasting future days
x_input = test_data[len(test_data)-100:].reshape(1,-1)
temp_input = list(x_input[0])
lst_output = []
days_to_forecast = st.sidebar.slider('Days to forecast', 1, 30, 10)

for _ in range(days_to_forecast):
    if len(temp_input) > 100:
        curr_input = np.array(temp_input[1:]).reshape((1, 100, 1))
        yhat = model.predict(curr_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
    else:
        curr_input = x_input.reshape((1, 100, 1))
        yhat = model.predict(curr_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

# --- STEP 6: VISUALIZATION ---
st.subheader(f"Analysis for {selected_name} ({ticker})")
col1, col2 = st.columns(2)

with col1:
    st.write("Historical Close Price (2015-2020)")
    fig1, ax1 = plt.subplots()
    ax1.plot(df.index, df['Close'])
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.write(f"Forecast for {days_to_forecast} Days")
    day_recent = np.arange(1, 101)
    day_pred = np.arange(101, 101 + days_to_forecast)
    fig2, ax2 = plt.subplots()
    ax2.plot(day_recent, scaler.inverse_transform(df1_scaled[-100:]), label="Recent")
    ax2.plot(day_pred, scaler.inverse_transform(lst_output), label="Forecast")
    ax2.legend()
    st.pyplot(fig2)

# --- STEP 7: RESULTS TABLE ---
st.subheader("Comparison Table (Test Data)")
valid_len = len(test_predict)
valid = df.iloc[-valid_len:].copy()
valid['Predictions'] = test_predict.flatten()
st.dataframe(valid[['Close', 'Predictions']].tail(15), use_container_width=True)