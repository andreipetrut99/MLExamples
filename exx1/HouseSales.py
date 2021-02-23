import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('kc_house_data.csv', delimiter=',')
print(df.info)

X = df.iloc[:, 3:]
y = df.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
print("R squared for test {}".format(model.score(X_test, y_test)))
print("MSE for test {}".format(mean_squared_error(y_test, y_pred)))
print("R squared for train {}".format(model.score(X_train, y_train)))
print("MSE for train {}".format(mean_squared_error(y_train, y_pred_train)))
