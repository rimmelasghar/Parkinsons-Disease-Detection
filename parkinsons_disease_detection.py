# making required imports
import numpy as np
import pandas as pd
import os,sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading the Dataset
df = pd.read_csv('parkinsons.data')
print(df.head())

# getting Features and Labels (target variable)
features = df.loc[:,df.columns!='name']
features = features.loc[:,features.columns!='status']
labels = df.loc[:,'status'].values

# number of value with value = 1
print(labels[labels == 1].shape)

# number of value with value = 0
labels[labels == 0].shape

# scale the features to between -1 and 1
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels

#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

# Training the model
model = XGBClassifier(random_state =1)
model.fit(x_train,y_train)

# prediction
y_pred = model.predict(x_test)

# Calculating the Accuracy
accuracy = accuracy_score(y_test,y_pred)*100
print(f"The result are {round(accuracy,2)}% accurate")