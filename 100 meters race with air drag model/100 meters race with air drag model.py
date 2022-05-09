import scipy
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from runner_data import get_runner_data as grd
from scipy.optimize import minimize


#Função v(t)
def v_of_t(a0,b,t):
    
    tan = np.tanh(np.sqrt(a0)*np.sqrt(b)*t)
    return (tan/np.sqrt(b/a0))
   
#Função x(t)
def x_of_t(a0,b,t):
    
    return np.log(np.cosh(t*np.sqrt(a0*b)))/b

#Função t(x)
def t_of_x(a0,b,x):
    
    tx = np.linspace(0,100,1000)
    xt = x_of_t(a0,b,tx)
    spl = interp1d(xt, tx, kind='cubic',assume_sorted=True)
    return spl(x)
    
#Função chi2
def chi2(p,tobs,xobs):
    
    a0 = p[0]
    b  = p[1]
    t_model = t_of_x(a0,b,xobs)
    dy2 = (tobs - t_model)**2
    return dy2.sum()




#Definição por chute 
a0= 8.5 
cd = 0.5 
p = 1.2
massa = 80
a = 1

#Corredor usando o chute
t = np.linspace(0,10,1000)                        
b = (1/(2*massa))*p*cd*a
v = v_of_t(a0,b,t)
x = x_of_t(a0,b,t)

plt.figure(1)
plt.plot(t,v, '-b', label='Função v(t) usando chute')
plt.xlabel('Instante t')
plt.ylabel('Velocidade')
plt.legend()

plt.figure(2)
plt.plot(t,x, '-r', label='Função x(t) usando chute')
plt.xlabel('Instante t')
plt.ylabel('Posição')
plt.legend()


plt.show() 

i = 0
for i in range(3):
    #Valores observados
    Corredor = grd()
    
    #Minimizando Chi2 para achar os melhores valores para a0 e b do corredor atual
    m = minimize(chi2, x0=[a0,b], args=(Corredor[1],Corredor[0]), method='Nelder-Mead', tol = 1.e-8)
    print( f'Corredor {i+1}' )
    print (f'a0  = {m.x[0]:.2f} m/s2.')
    print (f'B = {m.x[1]:.2f}')
    
    #Corredor usando as contantes minimizadas
    t = np.linspace(0,10,1000)
    vmin = v_of_t(m.x[0],m.x[1],t)
    xmin = x_of_t(m.x[0],m.x[1],t)
    
    
    #Comparando a velocidade limite
    #Podemos observar que Bolt atinge a maior velocidade maxima, e que apesar do powell ser o que possui maior acelerção,
    #ele tambem é o que mais sofre com o arrasto do ar, nota-se isso no ajuste do seu B e do A(Area da seção reta) encontrados.
    print('Comparação velocidade')
    vlim = np.sqrt(m.x[0]/m.x[1])
    print(vlim)
    print(vmin.max())


    #Area seção reta estimada
    Aesp = (m.x[1]*(2*80))/1.2*0.5
    print(f'Area estimada da seção Reta do Atleta = {Aesp} \n \n ')

    

    #Plotando os corredores
    plt.figure(3)
    plt.plot(t,vmin, '-b', label='Função v(t) ajustada')
    plt.xlabel('Instante t')
    plt.ylabel('Velocidade')
    plt.legend()

    plt.figure(4)
    plt.plot(t,xmin, '-r', label='Função x(t) ajustada')
    plt.xlabel('Instante t')
    plt.ylabel('Posição')
    plt.legend()
    
    plt.show()











