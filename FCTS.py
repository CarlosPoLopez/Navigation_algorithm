import numpy as np
import matplotlib.pyplot as plt
import os

#deltax=deltay
def FCTS(u, v, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F, plot_cada):

    u_new = np.copy(u)
    v_new = np.copy(v)

    #Configuración plots
    #plt.ion()
    plt.ioff()
    fig, ax = plt.subplots(figsize = (6,6))
    imagen = ax.imshow(u, cmap='magma', vmin=-1.5, vmax=1.5, origin='lower')
    plt.colorbar(imagen)

    carpeta_salida = 'Plots'
    os.makedirs(carpeta_salida, exist_ok=True)

    #Bucle
    for t in range(T):

        # Definimos todo menos los bordes, equivalente a range(1,N)
        u_c = u[1:N, 1:N]
        v_c = v[1:N, 1:N]

        # Laplacianos vectorizados 
        lap_u = (u[2:N+1, 1:N] + u[0:N-1, 1:N] + u[1:N, 2:N+1] + u[1:N, 0:N-1] - 4*u_c) / deltax**2
        lap_v = (v[2:N+1, 1:N] + v[0:N-1, 1:N] + v[1:N, 2:N+1] + v[1:N, 0:N-1] - 4*v_c) / deltax**2

        # Actualización de las matrices interiores 
        u_new[1:N, 1:N] = u_c + deltat * (epsilon * (u_c - u_c**3 - v_c + F) + Du * lap_u)
        v_new[1:N, 1:N] = v_c + deltat * (u_c - alpha * v_c + beta + Dv * lap_v)

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

        if t % plot_cada == 0:
            imagen.set_data(u)            
            ax.set_title(f'Paso de tiempo: {t}')
            #plt.pause(0.001)    
            ruta_archivo = os.path.join(carpeta_salida, f'frame_{t}.png')
            plt.savefig(ruta_archivo)
            print(f'Calculado t={t} y fotograma guardado.')          
    
    plt.ioff() # Apagamos el modo interactivo al final
    plt.show()    
    
    return u,v


