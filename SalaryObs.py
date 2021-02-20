import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary.csv', delimiter=',')

X = df['YearsExperience'].values
X = X.reshape(-1, 1)
y = df['Salary'].values

X_test, X_train, y_test, y_train = train_test_split(X, y, train_size=0.2)
# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)

plt.xlabel("Years of experience")
plt.ylabel("Salary")
# plt.show()
# correlation_matrix = df.corr()
# print(correlation_matrix)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# print("R squared for test: {}".format(model.score(X_test, y_test)))

def lin(x, a, b):
    return x * a + b


a = model.coef_[0]
b = model.intercept_
points = np.array([X.min(), X.max()])

plt.plot(points, lin(points, a, b), c = 'orange')
plt.show()
