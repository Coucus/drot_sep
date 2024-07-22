import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def Costfunction(X, Y, thetaX, thetaY):
    j = np.matrix(np.zeros(thetaX.shape))
    for i in range(len(X)):
        f = thetaX * X[i][0] + thetaY * X[i][1]
        j += np.power(f - Y[i], 2)
    cost = j / len(X)
    return cost

X_test = np.array([[1, 1],
                    [1, 2],
                    [1, 3],
                    [1,5],
                    [1,9],
                    [1,11],
                    [1,21]])
Y_test = np.array([1, 2, 3,5,8,10,13])
theta0_value = np.arange(-30, 30, 0.01)
theta1_value = np.arange(-30, 30, 0.01)
theta0, theta1 = np.meshgrid(theta0_value, theta1_value)

fig, ax = plt.subplots()
z = Costfunction(X_test, Y_test, theta0, theta1)
ax.contour(theta0, theta1, z)
ax.set_ylim(-50,50)
ax.set_xlim(-50,50)
plt.show()
