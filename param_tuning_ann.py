# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:34:16 2021

@author: Harsh Raj
"""


#Preprocessing

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset

df=pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13]
y = df.iloc[:, 13]

#creating dummy for object data
#dummy is not created for first to save memory
geog=pd.get_dummies(X["Geography"],drop_first=True)
sex=pd.get_dummies(X['Gender'],drop_first=True)

#concatenating dummy and dropping original var
X=pd.concat([X,geog,sex],axis=1)
X=X.drop(['Geography','Gender'],axis=1)

#preparing the training and test data
from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(X,y,test_size=0.3)

#Scaling the features
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_tr=ss.fit_transform(x_tr)
x_ts=ss.fit_transform(x_ts)

#Performing Hyper-parameter optimization for number of layers and activation function

#importing libraries
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid

#function for creating model with tuned hyperparameters
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=x_tr.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model

#creating the model for prediction with the function
model = KerasClassifier(build_fn=create_model, verbose=0)

#Hyperparameters for tuning
#layers = [[20], [40, 20], [45, 30, 15]]
layers = [(20), (40, 20), (45, 30, 15)]
activations = ['sigmoid', 'relu']

#preparing the dictionary of hyperparameters
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])

#initializing the gridsearch model
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

#fitting the training dataset in the model
grid_result = grid.fit(x_tr, y_tr)

print([grid_result.best_score_,grid_result.best_params_])

#Making the predictions
y_pred=grid.predict(x_ts)
y_pred=(y_pred>0.5)

#Evaluating the results
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_ts))







