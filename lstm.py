import streamlit as st
import datetime as dt
import os
import math
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

st.title('Cryptocurrency price prediction using LSTM model')
coins = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'POL-USD', 'DOGE-USD', 'AVAX-USD', 'DAI-USD',
         'MATIC-USD', 'SHIB-USD', 'UNI-USD', 'TRX-USD', 'ETC-USD', 'WBTC-USD', 'LEO-USD', 'LTC-USD', 'NEAR-USD', 'LINK-USD']

start_date = '2018-01-01'

# Forms For GUI
selected_pair = st.sidebar.radio(
    "Select Crypto Coins Pair", coins)
selected_coin = selected_pair.split('-')[0]
end_date = dt.date.today()
prediction_days = st.sidebar.number_input("Prediction Days", 7)


# Load dt (dt refers at data)

header_name = ['High', 'Low', 'Open', 'Close', 'Volume']
dt = web.DataReader(selected_pair,
                    'yahoo', start_date, end_date)
print(dt)

dt.shape
dt.isna()


dt = dt['Close']
fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(dt, label='Train', linewidth=2)
ax.set_ylabel('Price USD', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
ax.set_title('Weighted Price Graph', fontsize=16)
st.plotly_chart(fig, use_container_width=True)


# Split Training and Testing dt

testing_days = 60

dt_train = dt[:len(dt)-testing_days].values.reshape(-1, 1)
dt_test = dt[len(dt)-testing_days:].values.reshape(-1, 1)
#st.write('Train dt: ', dt_train.shape, 'Test dt: ', dt_test.shape)


scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(dt_train)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(dt_test)


SEQ_LEN = 100


def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)


def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(
    scaled_train, SEQ_LEN, train_split=0.95)

st.write("X_train: ", X_train.shape, "y_train: ",
         y_train.shape, "X_test: ", X_test.shape, "y_test", y_test.shape)

#st.write(X_train)


# Reshape trainX and testX into 3D

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

st.write("Shape of X_train: ", X_train.shape,
         "Shape of X_test: ", X_test.shape)

#st.write("trainX: ", X_train)


DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

regressor = Sequential()
regressor.add(LSTM(units=50,
                   activation='linear',
                   return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))
# Add dropout with a probability of 0.5
regressor.add(Dropout(.5))
regressor.add(LSTM(units=50,
                   return_sequences=False))
# Add dropout with a probability of 0.5
regressor.add(Dropout(.5))
regressor.add(Dense(25))
regressor.add(Dense(1))

regressor.compile(optimizer='adam', loss='mean_squared_error')


regressor_path = 'models/'+selected_coin + '_model.hdf5'

if os.path.exists(regressor_path):
    regressor = load_model(regressor_path)
else:
    regressor.fit(X_train, y_train, batch_size=32, epochs=25, verbose=1,
                  shuffle=False, validation_data=(X_test, y_test))
    regressor.save(regressor_path)

# Transformation to original form and making the predictions

predicted_price_test_dt = regressor.predict(X_test)

predicted_price_test_dt = scaler_test.inverse_transform(
    predicted_price_test_dt.reshape(-1, 1))

test_actual = scaler_test.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(predicted_price_test_dt, label='Predicted Test')
ax.plot(test_actual, label='Actual Test')
ax.set_ylabel('Price USD', fontsize=14)
ax.set_title('Predicted Test V/S Actual Test', fontsize=16)
ax.legend()
st.plotly_chart(fig, use_container_width=True)


predicted_price_train_dt = regressor.predict(X_train)

predicted_price_train_dt = scaler_train.inverse_transform(
    predicted_price_train_dt.reshape(-1, 1))

train_actual = scaler_train.inverse_transform(y_train.reshape(-1, 1))

fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(predicted_price_train_dt, label='Predicted train')
ax.plot(train_actual, label='Actual train')
ax.set_ylabel('Price USD', fontsize=14)
ax.set_title('Predicted train V/S Actual train', fontsize=16)
ax.legend()
st.plotly_chart(fig, use_container_width=True)


# RMSE and R2 score - Test dt
rmse_lstm_test = math.sqrt(mean_squared_error(
    test_actual, predicted_price_test_dt))
r2_scr_test = r2_score(
    test_actual, predicted_price_test_dt)


# RMSE and R2 score - Train dt
rmse_lstm_train = math.sqrt(mean_squared_error(
    train_actual, predicted_price_train_dt))
r2_scr_train = r2_score(
    train_actual, predicted_price_train_dt)

st.write('Test RMSE: ', rmse_lstm_test, 'Train RMSE: ', rmse_lstm_train)
st.write('Test R2 Score: ', r2_scr_test, 'Train R2 Score: ', r2_scr_train)

X_test_last_5_days = X_test[X_test.shape[0] - prediction_days:]

predicted_future_dt = []

for i in range(prediction_days):
    predicted_future_dt_x_test = regressor.predict(
        X_test_last_5_days[i:i+1])

    predicted_future_dt_x_test = scaler_test.inverse_transform(
        predicted_future_dt_x_test.reshape(-1, 1))
    # print(predicted_forecast_price_test_x)
    predicted_future_dt.append(
        predicted_future_dt_x_test)

predicted_future_dt = np.array(predicted_future_dt)
predicted_future_dt = predicted_future_dt.flatten()
predicted_price_test_dt = predicted_price_test_dt.flatten()

st.subheader('Next '+str(prediction_days)+' days Predicted Data')
st.write(predicted_future_dt)


predicted_test_concatenated = np.concatenate(
    (predicted_price_test_dt, predicted_future_dt))


fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(predicted_test_concatenated, label='Predicted Test')
ax.plot(test_actual, label='Actual Test')
ax.set_ylabel('Price USD', fontsize=14)
ax.set_title('Predicted Test V/S Actual Test', fontsize=16)
ax.legend()
st.plotly_chart(fig, use_container_width=True)
