#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 00:40:08 2019

@author: mamadcamzis
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Partie construire un reseau de neurones
# importer les modiules
import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialisation
classifier = Sequential()

# Ajouter une couche tout en initialisant la couche d'entrée ""input-dim"
classifier.add(Dense(units=6, activation='relu',
                     kernel_initializer='uniform', input_dim=11))

# Ajouter Dropout pour eviter le surapprentisaage
classifier.add(Dropout(rate=0.1))

# Une deuxieme couche cachée
classifier.add(Dense(units=6, activation='relu',
                     kernel_initializer='uniform'))

classifier.add(Dropout(rate=0.1))

# Ajouterd la couche de sortie
classifier.add(Dense(units=1, activation='sigmoid',
                     kernel_initializer='uniform'))


# compiler le modele
classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['accuracy'])

# Entrainer le modele
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Predire une observation seule
"""Pays: France

Score de credit: 600
Genre: Male
Age: 40
Durée depuis entrée dans la banque 3 ans
Balance: 60000€
Nombre de produits :2
Carted e credit ? oui
Membre actif? oui
Salaire estimé: 50000€"""

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 0, 40, 3,
                                                           60000, 2, 1, 1, 
                                                           50000]])))
new_prediction = (new_prediction > 0.5)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    
     # Initialisation
    classifier = Sequential()
    
    # Ajouter une couche tout en initialisant la couche d'entrée ""input-dim"
    classifier.add(Dense(units=6, activation='relu',
                         kernel_initializer='uniform', input_dim=11))
    
    # Une deuxieme couche cachée
    classifier.add(Dense(units=6, activation='relu',
                         kernel_initializer='uniform'))
    
    # Ajouterd la couche de sortie
    classifier.add(Dense(units=1, activation='sigmoid',
                         kernel_initializer='uniform'))
    
    
    # compiler le modele
    classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                       metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, 
                             epochs=100)
precisions = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

moyenne = precisions.mean()
std = precisions.std()
    
# Partie 4


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    
     # Initialisation
    classifier = Sequential()
    
    # Ajouter une couche tout en initialisant la couche d'entrée ""input-dim"
    classifier.add(Dense(units=6, activation='relu',
                         kernel_initializer='uniform', input_dim=11))
    
    # Une deuxieme couche cachée
    classifier.add(Dense(units=6, activation='relu',
                         kernel_initializer='uniform'))
    
    # Ajouterd la couche de sortie
    classifier.add(Dense(units=1, activation='sigmoid',
                         kernel_initializer='uniform'))
    
    
    # compiler le modele
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', 
                       metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, 
                             epochs=100)
parameters = {"batch_size": [25, 32], "epochs": [100, 500],
              "optimizer": ["adam", "rmsprop"] }
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters,
                           scoring="accuracy", cv=10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)