import matplotlib.pyplot as plt
import numpy as np

ore = np.array([2, 2.5, 3, 3.4, 4.2, 5.1, 5.8, 6, 6.5, 7, 7.3, 8, 8.6, 9, 10])
indice_performanta = np.array([30, 34, 39, 46, 55, 61, 58, 66, 75, 80, 88, 91, 92, 94, 99])

ore_mean = np.mean(ore)
performanta_mean = np.mean(indice_performanta)

covariance = np.sum((ore - ore_mean)*(indice_performanta - performanta_mean))
variance = np.sum(np.square(ore - ore_mean))

a = covariance / variance
b = performanta_mean - (a * ore_mean)

plt.scatter(ore, indice_performanta)

#fit function
f = lambda x: a*x + b


x = np.array([min(ore), max(ore)])

plt.plot(x, f(x), c="orange")
plt.xlabel('Număr ore dormite(var.independentă)')
plt.ylabel('Indice Performanță(var.dependentă)')
plt.legend()
plt.show()