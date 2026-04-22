import matplotlib.pyplot as plt
import numpy as np

def f(u, F):
    return u - u**3 + F

def g(u, alpha, beta):
    return (u + beta)/alpha

u = np.linspace(-1, 1)

F = 0
""" alpha = 8
beta = -0.33 """


alpha = 5
beta = 0.1


f = f(u, F)
g = g(u, alpha, beta)

plt.figure(figsize=(6,6))
plt.plot(u, f, color='blue')
plt.plot(u, g, color='orange')
plt.xlabel('u')
plt.ylabel('v')
plt.show()
