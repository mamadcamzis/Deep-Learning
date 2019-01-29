#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:46:10 2019

@author: mamadcamzis
"""

#   Méga cas d'étude - Modèle hybride de Deep Learning 


#   Partie 1 -  Détecter la fraude aevc une carte auto adaptative

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import les donnee
dataset  = pd.read_csv("Credit_Card_Applications.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Changement d'echelle
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Entrainement de Som
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# Visualisation des resultats
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
colors = ["r", "g"]

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markeredgecolor=colors[y[i]],
    markerfacecolor="None", markersize=10, markeredgewidth=2)
    
    
show()

# Detecter les fraudes

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2, 3)], mappings[(3,1)]),
                        axis=0)
frauds = sc.inverse_transform(frauds)

#   Partie 2 -  Passer du non-supervisé au supervisé

# Création de la matrice de variables

customers = dataset.iloc[:, 1:].values
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)



# Partie construire un reseau de neurones
# importer les modiules
import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialisation
classifier = Sequential()

# Ajouter une couche tout en initialisant la couche d'entrée ""input-dim"
classifier.add(Dense(units=2, activation='relu',
                     kernel_initializer='uniform', input_dim=15))

# Ajouter Dropout pour eviter le surapprentisaage
classifier.add(Dropout(rate=0.1))



# Ajouterd la couche de sortie
classifier.add(Dense(units=1, activation='sigmoid',
                     kernel_initializer='uniform'))


# compiler le modele
classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['accuracy'])

# Entrainer le modele
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the Test set results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]

