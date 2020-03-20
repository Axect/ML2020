from functools import singledispatch
import numpy as np

class Vector:
    def __init__(self, arr):
        self.arr = arr

class Matrix:
    def __init__(self, m):
        self.m = m

@singledispatch
def mean(x):
    print("qwe")

@singledispatch
def var(x):
    pass

@mean.register(Vector)
def _(x):
    pass

@mean.register(Matrix)
def _(x):
    pass

def cov(x, y):
    pass

def cov_mat(m):
    pass

a = Vector([1,2,3,4,5])
b = Vector([5,4,3,2,1])
print(mean(a))
print(var(a))
print(cov(a,b))

m = Matrix(np.matrix("1 8;2 6;3 4;4 2"))
print(mean(m))
print(var(m))
print(cov_mat(m))
