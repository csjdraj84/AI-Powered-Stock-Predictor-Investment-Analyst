import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

class LSTMModel:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def prepare_data(self, data):
        """Prepare data for LSTM training."""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        x_train, y_train = [], []
        for i in range(self.lookback, len(scaled_data)):
            x_train.append(scaled_data[i-self.lookback:i, 0])
            y_train.append(scaled_data[i, 0])
            
        return np.array(x_train), np.array(y_train)

    def build_model(self, input_shape):
        """Build and compile a lighter LSTM model for speed."""
        model = Sequential([
            LSTM(units=32, return_sequences=False, input_shape=input_shape),
            Dropout(0.1),
            Dense(units=16),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train(self, data, epochs=5, batch_size=64):
        """Train the LSTM model with optimized parameters."""
        # Ensure data is 1D
        if len(data.shape) > 1:
            data = data.flatten()
            
        # Use only the last 500 days for training to speed up
        if len(data) > 500:
            data = data[-500:]
            
        x_train, y_train = self.prepare_data(data)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        self.build_model((x_train.shape[1], 1))
        # Use a smaller number of epochs and larger batch size for speed
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return self.model

    def predict_future(self, data, days_to_predict=30):
        """Predict future stock prices."""
        # Ensure data is 1D
        if len(data.shape) > 1:
            data = data.flatten()
            
        last_val = data[-self.lookback:]
        last_val_scaled = self.scaler.transform(last_val.reshape(-1, 1))
        
        future_predictions = []
        current_batch = last_val_scaled.reshape((1, self.lookback, 1))
        
        for _ in range(days_to_predict):
            pred = self.model.predict(current_batch, verbose=0)[0]
            future_predictions.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            
        return self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    def get_historical_predictions(self, data):
        """Get predictions for the historical data to show model fit."""
        # Ensure data is 1D
        if len(data.shape) > 1:
            data = data.flatten()
            
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        x_test = []
        for i in range(self.lookback, len(scaled_data)):
            x_test.append(scaled_data[i-self.lookback:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        predictions = self.model.predict(x_test, verbose=0)
        return self.scaler.inverse_transform(predictions)
