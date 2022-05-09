import math
import scipy
import numpy 
from scipy.integrate import quad
from numpy import linspace
from math import sqrt
import matplotlib.pyplot as plt

def func_E(x, omega_m):
    e = math.sqrt(omega_m * (1+x)**3 + 1 - omega_m)
    return e

def func_inte(x,omega_m):
    return (1/((1+x)*func_E(x,omega_m)))


def idade(omega_m):
    h0 = 67.8
    integral = quad(func_inte, 0, numpy.inf, args=(omega_m),epsabs=1.49e-12)
    T = ((1/h0)*integral[0])
    return T


omegam = 0.308
resp = idade(omegam)
print (resp)
Omega_var = linspace(0.15,0.45,100)
resp = list(map(idade, Omega_var))
plt.plot(Omega_var,resp, '-b')
plt.xlabel('OmegaM')
plt.ylabel('Idade')
plt.show()

#Pela analise do grafico,
#percebe-se que a medida que o parametro de densidade fica menor
#A idade do universo aumenta.
