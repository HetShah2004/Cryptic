import yfinance as yf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Get the cryptocurrency data from Yahoo Finance API
symbol = 'BTC-USD'
crypto_data = yf.Ticker(symbol).history(period='max')

# Filter the necessary columns and reset the index
crypto_data = crypto_data[['Close']]
crypto_data = crypto_data.reset_index()

# Create a variable for the number of days to predict
days_to_predict = 30

# Prepare the training data for the SVR model
X_train = np.array(range(1, len(crypto_data) + 1)).reshape(-1, 1)
y_train = np.array(crypto_data['Close'])

# Scale the training data
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

# Create the SVR model and fit the data
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train.ravel())

# Prepare the data to predict future prices
X_test = np.array(range(len(crypto_data) + 1, len(crypto_data) + days_to_predict + 1)).reshape(-1, 1)

# Scale the test data
X_test_scaled = sc_X.transform(X_test)

# Predict the future prices
y_pred_scaled = svr.predict(X_test_scaled)

# Rescale the predicted prices (reshaping the array to 2D)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Visualize the data and the predicted prices
st.subheader("Cryptocurrency Price Prediction using Support Vector Regression (SVR)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(crypto_data['Close'], label='Actual Price')

# Create the full plot data correctly
predicted_plot_data = np.empty((len(crypto_data) + len(y_pred), 1))
predicted_plot_data[:] = np.nan
predicted_plot_data[len(crypto_data):] = y_pred

ax.plot(predicted_plot_data, label='Predicted Price')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)