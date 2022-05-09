import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def func(t, y, q, m, E, B):
    f = np.empty_like (y)

    f[0] = y[1]
    f[1] = (q/m)*B*y[3]
    
    f[2] = y[3]
    f[3] = (q/m)*(E-(B*y[1]))

    return f

def main():
    
    q = 1
    m = 1
    E = 1
    B = 1
    
    #y0 = [y, vy,z,vz]
    #Condição inicial: Particula em repouso na origem do referencial

    y0 = [0,0,0,0]
    t_e = np.linspace(0,50,1000)
    t_s = (t_e.min(), t_e.max())
    s   = solve_ivp(func, t_span = t_s, y0 = y0, args =(q,m,E,B),t_eval=t_e)

    plt.figure(1)
    plt.plot (s.y[0], s.y[2], '-b')
    plt.plot(t_e,np.sqrt(((s.y[0]-t_e)**2)+(s.y[2]-1)**2),'-r')
    plt.show ()
    return True
    '''
    R = sqrt(((t-sen(t)-t)^2)+(1-cos(t)-1)^2)
    R = sqrt((-sen(t)^2)+(-cos(t))^2)
    R = sqrt((sen(t)^2)+(cos(t))^2)
    Com a dica temos:
    R = sqrt(sen(t)^2 + cos(t)^2) = 1
    R é o raio do sistema
    '''
    

if __name__ == "__main__":
    main()

