#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 00:10:13 2019

@author: mamadcamzis
"""

# Reccureent Neural Network
# Librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Partie 1 - Preparation de donnees

# Jeu d'entraianement
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train[["Open"]].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creation de la structure avec 60 timesteps
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Partie 2 - Construction du RNN
# Librairie

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True, 
                   input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 2e couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 3e couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 4e couche LSTM + Dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Couche de sortie
regressor.add(Dense(units=1))

# Compilation
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Enrainement
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# Partie 3 - Predictions et viswualisation

# Donnée de 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test[["Open"]].values

# Predictions pour 2017
dataset_total = pd.concat((dataset_train['Open'],
                           dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualisation des resulats
plt.plot(real_stock_price, color='red', 
         label="Prix réel de l'action Google")
plt.plot(predicted_stock_price, color='green',
         label="Prix prédit de l'action Google")
plt.title("Prédiction de l'action google")
plt.xlabel("Jour")
plt.legend()
plt.ylabel("Prix de l'action")
plt.show()

# =============================================================================
#     y_train = []
#     for i in range(60, 1258):
#         y_train.append(training_set_scaled[i, 0])
#     y_train = np.array(y_train)
# =============================================================================
# =============================================================================
# # Entrainement plusieurs variables
#     X_train = []
#     for variable in range(0, 2):
#         X = []
#         for i in range(60, 1258):
#             X.append(training_set_scaled[i-60:i, variable])
#         X, np.array(X)
#         X_train.append(X)
#     X_train, np.array(X_train)
# =============================================================================
