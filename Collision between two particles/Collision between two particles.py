import sys
import scipy
from scipy.optimize import root_scalar
import numpy as np
import matplotlib.pyplot as plt

#Para a primeira particula temos:
# x1(t) = v0*t
# y1(t) = (1/2)*a*t**2
# onde v0 = 10 e a = 5
# logo :
# x1(t) = 10*t
# y1(t) = (1/2)*5*t**2

#Para a segunda particula temos:
# x2(t) = xm * tanh(t/T)
# y2(t) = y0 * e**(t/T)
# onde xm = 100 T = 5
# logo :
# x2(t) = 100*tanh(t/5)
# y2(t) = y0*e**(t/5)

#A colisão ocorre quando x1(t) = x2(t) e y1(t) = y2(t)
# 10*t = 100*tanh(t/5)
# (1/2)*5*t**2 = y0*e**(t/5)
# Isolando y0:
# y0 = ((1/2)*5*t**2)/e**(t/5)




#Tentei fazer uma função que retornasse x e y como uma tupla
#Porém tive dificuldades na hora de usar a root_scalar
#Como o problema era justamente relacionado as tuplas, resolvi separar as funções
#Funções que retornam x e y da primeira particula:
def x1(t):
#   v0 = 10
    x  = 10*t
    return x

def y1(t):
#   a  = 5
    y  = (1/2)*5*t**2
    return y
      
#Funções que retornam x e y da segunda particula

def x2(t):
#   xm = 100
    x = 100*np.tanh(t/5)
    return x

def y2(t,y0):
#   T  = 5
    y = y0*np.e**(t/5)
    return y
    
#Função que retorna y0

def y0 (t) :
    return ((1/2)*5*t**2)/np.e**(t/5)

#Função que usamos como parametro para root_scalar
#x1(t)=x2(t) -> x1(t)-x2(t) = 0
def diferença(t , x1 , x2):
    return x1(t) - x2(t)

#Solução da letra a)

solucao = root_scalar(diferença , args=(x1,x2), method='brentq', bracket=[sys.float_info.min, sys.float_info.max])
print(f"t = {solucao.root}\nyo = {y0 (solucao.root)}")

#Solução da letra b)

t = np.linspace (0, 30, 30)

plt.figure(1)
plt.plot(x1(t),y1(t), '-b', label='Particula 1') 
plt.plot(x2(t),y2(t,y0(t)), '-r', label='Particula 2')
plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.legend()

plt.figure(2)
plt.plot(t,x1(t), '-b', label='Particula 1') 
plt.plot(t,x2(t), '-r', label='Particula 2')
plt.xlabel('Instante t')
plt.ylabel('X(t)')
plt.legend()
plt.show()

