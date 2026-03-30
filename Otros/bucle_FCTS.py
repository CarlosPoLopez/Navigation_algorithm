import numpy as np
import matplotlib.pyplot as plt

#deltax=deltay
def FCTS(deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F):
    
    #Matrices u y v
    u = np.zeros([N+1, N+1])
    v = np.zeros([N+1, N+1])
    u_new = np.zeros([N+1, N+1])
    v_new = np.zeros([N+1, N+1])

    #Condición inicial
    u[1:6, 1:6] = 1.0

    #Configuración plots
    plt.ion()
    fig, ax = plt.subplots()
    imagen = ax.imshow(u, cmap='magma', vmin=-1.5, vmax=1.5, origin='lower')
    plt.colorbar(imagen)

    #Bucle
    for t in range(T):
        for i in range(1,N):
            for j in range(1,N):
                u_new[i,j] = u[i,j] + deltat*(epsilon*(u[i,j] - u[i,j]**3 - v[i,j] + F) + Du*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])/deltax**2)
                v_new[i,j] = v[i,j] + deltat*(u[i,j] - alpha*v[i,j] + beta + Dv*(v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4*v[i,j])/deltax**2)
        #Condiciones de Neumann (Zero-Flux)
        u_new[0, :] = np.copy(u_new[1, :])   
        u_new[-1, :] = np.copy(u_new[-2, :]) 
        u_new[:, 0] = np.copy(u_new[:, 1])   
        u_new[:, -1] = np.copy(u_new[:, -2]) 

        v_new[0, :] = np.copy(v_new[1, :])
        v_new[-1, :] = np.copy(v_new[-2, :])
        v_new[:, 0] = np.copy(v_new[:, 1])
        v_new[:, -1] = np.copy(v_new[:, -2])

        #Guardo u y v
        u = np.copy(u_new)
        v = np.copy(v_new)

        if t % 50 == 0:
            imagen.set_data(u)            
            ax.set_title(f'Paso de tiempo: {t}')
            plt.pause(0.001)              
    
    plt.ioff() # Apagamos el modo interactivo al final
    plt.show()    
    
    return u,v

deltat = 0.002
deltax = 0.09 
N = 100
T = 3000
epsilon = 10
alpha = 8.0
beta = -0.33
Du = 0.05
Dv = 2.0
F = 0

prueba = FCTS(deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F)
