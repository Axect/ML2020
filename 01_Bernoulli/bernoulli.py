import numpy as np
import sys
from random import random

def main(n):
    d = gen_data(n)
    mu_ml = mle(d)
    print("mle: ", mu_ml)
    bayes = Beta(2,2)
    bayes.update(d)
    print(bayes)
    print("optimal: ", bayes.find_optimal())

def gen_data(n):
    return np.random.randint(2, size=n)

def mle(d):
    return np.sum(d) / len(d)

class Beta:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def update(self, d):
        m = (d == 1).sum()
        l = len(d) - m
        self.a += m
        self.b += l

    def __str__(self):
        return "Beta(" + str(self.a) + ", " + str(self.b) + ")"

    def find_optimal(self):
        return (self.a - 1) / (self.a + self.b - 2)

if __name__=="__main__":
    main(int(sys.argv[1]))
