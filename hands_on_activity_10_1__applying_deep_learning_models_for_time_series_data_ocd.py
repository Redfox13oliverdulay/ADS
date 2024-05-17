# -*- coding: utf-8 -*-
"""Hands-on Activity 10.1 _Applying Deep Learning Models for Time-series Data_OCD.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jYVHMt6IelY39vU0Va4BSBND8HNh_NjW

**PSMDSSC 104-PSMDS001 - Advanced Data Science**
<br>
**Hands-on Activity 10.1 _Applying Deep Learning Models for Time-series Data**
<br>
<br>
**OLIVER DULAY**

**1.APPLICATION OF MLP, CNN AND LSTM IN A UNIVARIATE DATASET**
<BR>
<BR>
The dataset is the **AIR QUALITY MEASUREMENT**, commonly gathered using sensors in urban environments or near industrial areas with the follwoing key components:
<br>
**CO(GT):** Carbon Monoxide levels in mg/m³. CO is a common pollutant emitted from vehicles and combustion processes.
<br>
**NOx, NO2:** Nitrogen oxides levels, which are harmful gases also primarily emitted from vehicles and industrial activities.
<br>
**C6H6(GT):** Benzene levels, a carcinogenic pollutant.
T: Temperature in °C.
<br>
**RH:** Relative Humidity in %.
<br>
**AH:** Absolute Humidity.
<br>
**PT08.S1 (CO), PT08.S2 (NMHC)**: which indicate readings from various chemical sensors.
<br>
<br>
**"CO(GT)"** is chosen indicator for Forecasting for the following reasons:
<br>
**Environmental Impact:** CO is a significant environmental pollutant with direct health implications, particularly in urban settings. Forecasting CO levels can help in pollution management and public health advisories.
<br>
**Regulatory and Public Interest:** There's often specific interest from environmental agencies and the public in understanding and predicting CO levels due to its direct impact on air quality and health.
<br>
**Data Availability and Reliability:** CO levels are typically measured continuously, providing a robust dataset that can be used for accurate forecasting.
<br>
**Relevance in Time Series Forecasting:** CO concentrations can exhibit clear patterns such as daily variations due to traffic flow and seasonal variations influenced by changes in weather and heating practices. This makes it a suitable candidate for time series analysis and forecasting using machine learning models.
<br>
"""

import streamlit as st
import pandas as pd



# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, GlobalAveragePooling1D, LSTM

"""**LOAD AND PROCESS THE DATASET**"""

# Load the dataset
# Set the path to the CSV file (adjust the path as necessary for your environment)
data_path = 'https://drive.google.com/file/d/1X1t6EEAcjwpcoJMMtgBu51kwjuF1YEuQ/view?usp=drive_link'

@st.cache  # This decorator caches the data so the file isn't reloaded on every interaction
def load_data(path):
    data = pd.read_csv(path, sep=';')
    data.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)
    cols_to_convert = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
    data[cols_to_convert] = data[cols_to_convert].apply(lambda x: x.str.replace(',', '.').astype(float))
    return data

data = load_data(data_path)

# You can display the DataFrame in Streamlit using:
st.write(data)

data = pd.read_csv(data_path, sep=';')
data.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)
cols_to_convert = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
data[cols_to_convert] = data[cols_to_convert].apply(lambda x: x.str.replace(',', '.').astype(float))

# Parse DateTime and clean rows with unparseable dates
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
data = data.dropna(subset=['DateTime'])  # Drop rows where DateTime could not be parsed

data.set_index('DateTime', inplace=True)
data.drop(columns=['Date', 'Time'], inplace=True)

# Interpolate missing values in the data
data_interpolated = data.interpolate(method='time')

"""**VISUALIZATION OF THE DATASET**"""

# Plotting the data
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
data_interpolated['CO(GT)'].plot(title='CO(GT) Time Series')
plt.xlabel('Date')
plt.ylabel('CO(GT) Concentration')

plt.subplot(2, 2, 2)
data_interpolated['CO(GT)'].hist(bins=30)
plt.title('CO(GT) Distribution')
plt.xlabel('CO(GT) Concentration')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
data_interpolated.boxplot(column=['CO(GT)'])
plt.title('CO(GT) Box Plot')

plt.subplot(2, 2, 4)
pd.plotting.autocorrelation_plot(data_interpolated['CO(GT)'])
plt.title('CO(GT) Autocorrelation Plot')

plt.tight_layout()
plt.show()

"""**STATISTICAL SUMMARIES**"""

# Calculate statistical summary for CO(GT)
co_gt_stats = data_interpolated['CO(GT)'].describe()

# Calculate statistical summary for the entire dataset
full_stats = data_interpolated.describe()

# Display the statistics
print("Statistical Summary for CO(GT):")
print(co_gt_stats)
print("\nStatistical Summary for Entire Dataset:")
print(full_stats)

"""**NORMALIZE THE DATA**"""

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_interpolated)
data_normalized = pd.DataFrame(data_normalized, columns=data_interpolated.columns, index=data_interpolated.index)

"""**DEFINE FUNCTIONS**"""

# Define function to create sequences
def create_sequences(data, target_column, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, target_column]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

"""**PREPARE SEQUENCES**"""

# Prepare sequences
n_steps = 24
target_column = data_normalized.columns.get_loc('CO(GT)')
X, y = create_sequences(data_normalized.values, target_column, n_steps)
n_train = int(len(X) * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

"""**DEFINE AND PLOT MLP MODEL**"""

# Define MLP model
mlp_model = Sequential([
    Flatten(input_shape=(n_steps, len(data_normalized.columns))),
    Dense(100, activation='relu'),
    Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mse')
mlp_history = mlp_model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Plot MLP training and validation loss
plt.figure(figsize=(10, 4))
plt.plot(mlp_history.history['loss'], label='Train Loss')
plt.plot(mlp_history.history['val_loss'], label='Validation Loss')
plt.title('MLP Training and Validation Loss')
plt.legend()
plt.show()

"""**DEFINE AND PLOT CNN MODEL**"""

# Define CNN model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, len(data_normalized.columns))),
    GlobalAveragePooling1D(),
    Dense(50, activation='relu'),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse')
cnn_history = cnn_model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Plot CNN training and validation loss
plt.figure(figsize=(10, 4))
plt.plot(cnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title('CNN Training and Validation Loss')
plt.legend()
plt.show()

"""**DEFINE AND PLOT LSTM MODE**L"""

# Define LSTM model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, len(data_normalized.columns))),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_history = lstm_model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Plot LSTM training and validation loss
plt.figure(figsize=(10, 4))
plt.plot(lstm_history.history['loss'], label='Train Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training and Validation Loss')
plt.legend()
plt.show()

"""**EVALUATE THE MODELS**"""

# Evaluate models
mlp_loss = mlp_model.evaluate(X_test, y_test, verbose=0)
cnn_loss = cnn_model.evaluate(X_test, y_test, verbose=0)
lstm_loss = lstm_model.evaluate(X_test, y_test, verbose=0)

print(f'MLP Loss: {mlp_loss}, CNN Loss: {cnn_loss}, LSTM Loss: {lstm_loss}')

"""**COMPARISON OF THE THREE MODELS**"""

# Plot training and validation loss for each model side by side
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# MLP plot
axes[0].plot(mlp_history.history['loss'], label='Train Loss')
axes[0].plot(mlp_history.history['val_loss'], label='Validation Loss')
axes[0].set_title('MLP Training and Validation Loss')
axes[0].legend()

# CNN plot
axes[1].plot(cnn_history.history['loss'], label='Train Loss')
axes[1].plot(cnn_history.history['val_loss'], label='Validation Loss')
axes[1].set_title('CNN Training and Validation Loss')
axes[1].legend()

# LSTM plot
axes[2].plot(lstm_history.history['loss'], label='Train Loss')
axes[2].plot(lstm_history.history['val_loss'], label='Validation Loss')
axes[2].set_title('LSTM Training and Validation Loss')
axes[2].legend()

plt.tight_layout()
plt.show()

"""**2.APPLICATION OF MLP, CNN AND LSTM IN A MULTIVARIATE DATASET**

The "**Household Electric Power Consumption**" dataset is a multivariate time series dataset that captures the power consumption in a single household over a period from December 2006 to November 2010 (about 4 years). The data is sampled at a one-minute rate. Here’s a brief description of the key variables included in the dataset:
<BR>
<BR>
**Global Active Power:** The total active power consumed by the household (kilowatts). This is the primary variable of interest in many analyses, as it reflects the amount of power actively used for performing tasks.
<BR>
**Global Reactive Power:** The total reactive power consumed by the household (kilovolt-amperes). Reactive power is used in power systems to transmit power across the network and is not directly useful for household tasks but necessary for efficient electrical systems operation.
<BR>
**Voltage:** The average voltage (volts) measured in the household during the minute.
<BR>
**Global Intensity:** The total current intensity (amperes) used by the household. This is another measure of how much power is being used.
<BR>
**Sub Metering 1:** The energy sub-metered in the kitchen, covering large appliances like an oven, microwave, toaster, and kettle.
<BR>
**Sub Metering 2:** The energy sub-metered in the laundry room, covering devices like a washing machine, dryer, refrigerator, and light.
<BR>
**Sub Metering 3:** The energy sub-metered in an electric water heater and an air conditioner.
<BR>
<BR>
These measurements help in understanding the different ways energy is consumed in a household, including baseline and peak consumption times. The data can be used for various purposes, such as energy usage pattern analysis, forecasting energy consumption, and devising strategies for energy savings and efficient appliance use.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split

"""**LOAD THE DATASET**"""

# Load data
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
df = pd.read_csv(data_url, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                 infer_datetime_format=True, na_values='?', low_memory=False)

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Select necessary columns
columns_to_use = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
df = df[columns_to_use]

"""**VISUALIZATION**"""

# Plot sample data
df[['Global_active_power', 'Global_reactive_power', 'Voltage']].iloc[:1000].plot()
plt.title('Sample Data Plot')
plt.ylabel('Measurements')
plt.xlabel('Time (in intervals)')
plt.show()

"""**STATISTICAL SUMMARIES**"""

# Display statistical summary
print(df.describe())

"""**NORMALIZE THE DATA**"""

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

"""**TRAINING THE DATASET**"""

# Create dataset for training
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 1:]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Split data
train_size = int(len(scaled_data) * 0.67)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

look_back = 10
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape input for models
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

"""**DEFINE AND COMPILE THE MODEL**"""

# Define and compile MLP, CNN, LSTM models
mlp_model = Sequential([
    Flatten(input_shape=(look_back, len(columns_to_use)-1)),
    Dense(50, activation='relu'),
    Dense(1)
])
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, len(columns_to_use)-1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])
lstm_model = Sequential([
    LSTM(50, input_shape=(look_back, len(columns_to_use)-1)),
    Dense(1)
])

mlp_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

"""**FITTING AND PLOTTING THE MODEL**"""

# Fit models and plot training history
def fit_and_plot(model, name):
    history = model.fit(X_train, y_train, epochs=10, batch_size=70, validation_data=(X_test, y_test))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    return history

mlp_history = fit_and_plot(mlp_model, 'MLP')
cnn_history = fit_and_plot(cnn_model, 'CNN')
lstm_history = fit_and_plot(lstm_model, 'LSTM')

"""**PLOT COMPARISON**"""

# Plot training history
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(mlp_history.history['loss'], label='Train Loss')
plt.plot(mlp_history.history['val_loss'], label='Validation Loss')
plt.title('MLP Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(cnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(lstm_history.history['loss'], label='Train Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
