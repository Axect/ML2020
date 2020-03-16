import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Simple Linear Regression
# ==============================================================================
# x: D x 1
# Phi: Linear design matrix
def DesignMatrix(x):
    _, Phi = np.meshgrid(x, x)
    return np.matrix(Phi)

def weight(Phi, t):
    return np.matmul(np.linalg.pinv(Phi), t.T)

def linear_regression(W, X):
    return np.matmul(X, W.T)

# ==============================================================================
# Data Generation
# ==============================================================================
df = pd.read_csv("data.csv")
X = np.array(df["X"])
t = np.array(df["Y"])

Phi = DesignMatrix(X)
W = weight(Phi, t)
Y = linear_regression(W, X)

plt.scatter(X, t)
plt.plot(X, Y)
plt.savefig("plot.png")