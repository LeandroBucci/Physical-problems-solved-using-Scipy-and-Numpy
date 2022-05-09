import numpy as np
import matplotlib.pyplot as plt

def retorna_bebado(prob1,prob2):
    nwalkers = 100000
    nsteps = 101
    bebado = np.random.choice(np.array ([ -1 , 1]), p=[prob1,prob2], size=(nsteps,nwalkers))
    bebado [0 ,:] = 0
    bebado = np.cumsum(bebado,axis = 0)
    return bebado

bebado1 = retorna_bebado(0.5,0.5)
bebado2 = retorna_bebado(0.4,0.6)
bebado3 = retorna_bebado(0.3,0.7)
bebado4 = retorna_bebado(0.2,0.8)
bebado5 = retorna_bebado(0.1,0.9)

plt.figure(1)
plt.hist(bebado1[-1],bins =40)
plt.hist(bebado2[-1],bins =40)
plt.hist(bebado3[-1],bins =40)
plt.hist(bebado4[-1],bins =40)
plt.hist(bebado5[-1],bins =40)
plt.show()
