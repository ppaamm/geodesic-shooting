import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x
#    return x**(3)

def f_inv(x):
    return x
#    if x < 0: return - ((-x)**(1./3))
#    else: return x**(1./3)

def analogy(a, b, c):
    return f_inv(f(c) + f(b) - f(a))

#A = np.array([0, 0])
#B = np.array([2, 0])
#C = np.array([0, 3])
#
#D = analogy(A, B, C)

A = 0
B = 5

X = np.linspace(-20, 20, num=200)
plt.plot(X, [analogy(A, B, x) for x in X])
