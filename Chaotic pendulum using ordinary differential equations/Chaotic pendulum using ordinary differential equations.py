import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft

def fun(t,y,q,omega_0,b):
    f = np.empty_like(y)

    f[0] = y[1]
    f[1] = -q*y[1] - np.sin(y[0]) + b*np.cos(omega_0*t)
    
    return f


def main():
    #Parametros
    q       = 0.5
    b       = [0.9,1.15]
    omega_0 = 2/3
    theta   = np.radians(90)
    omega   = 0
    y0      = [theta, omega]
    N       = 100000
    te      = np.linspace (0,1000, N)
    ts      = (te.min(), te.max())
    s       = []
    nu_0    = omega_0/2*np.pi
    
    
    #Plot usando os 2 valores de b
    plt.xlim(0,200)
    plt.ylim(-20,20)
    s.append(solve_ivp (fun, t_span=ts, y0=y0, t_eval=te, args=(q,omega_0,b[0]), rtol=1.e-12, atol=1.e-12))
    plt.plot(te,s[0].y[0],'-b')
    s.append(solve_ivp (fun, t_span=ts, y0=y0, t_eval=te, args=(q,omega_0,b[1]), rtol=1.e-12, atol=1.e-12))
    plt.plot(te,s[1].y[0],'-r')
    plt.show()

 

    #Fft
    af1   = fft (s[0].y[0])[: N//2]
    af2   = fft (s[1].y[0])[: N//2]
    freq1 = np.linspace(0,af1.max()/nu_0,N//2)
    freq2 = np.linspace(0,af2.max()/nu_0,N//2)
    cn1   = 2 * np.abs (af1) / N
    cn2   = 2 * np.abs (af2) / N
    
    plt.figure(1)
    plt.plot (freq1, cn1, '-b')
    
    
    plt.figure(2)
    plt.plot (freq2, cn2, '-r')
    
    plt.show ()



if __name__ == "__main__":
    main()
