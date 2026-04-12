import numpy as np
import matplotlib.pyplot as plt
import os

#deltax=deltay
def FCTS(u, v, u_max, v_max, u_min, v_min, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F, plot_cada, fase):

    u_new = np.copy(u)
    v_new = np.copy(v)

    #Configuración plots
    #plt.ion()
    plt.ioff()
    fig, ax = plt.subplots(figsize = (6,6))
    imagen = ax.imshow(u, cmap='magma', vmin=-1.5, vmax=1.5, origin='lower')
    #plt.colorbar(imagen)

    #Pintar laberintos
    overlay_muro = np.zeros((N+1, N+1, 4))
    es_pared = (F < 0)
    overlay_muro[es_pared] = [0.8, 0.8, 0.8, 0.8]
    ax.imshow(overlay_muro, origin='lower')

    if fase == 1:
        #carpeta_salida = 'Plots_1.2'
        carpeta_salida = 'LAB_ida'
        os.makedirs(carpeta_salida, exist_ok=True)

        #Bucle
        for t in range(T):

            # Definimos todo menos los bordes, equivalente a range(1,N)
            u_c = u[1:N, 1:N]
            v_c = v[1:N, 1:N]
            F_c = F[1:N, 1:N]

            # Laplacianos vectorizados 
            lap_u = (u[2:N+1, 1:N] + u[0:N-1, 1:N] + u[1:N, 2:N+1] + u[1:N, 0:N-1] - 4*u_c) / deltax**2
            lap_v = (v[2:N+1, 1:N] + v[0:N-1, 1:N] + v[1:N, 2:N+1] + v[1:N, 0:N-1] - 4*v_c) / deltax**2

            # Actualización de las matrices interiores 
            u_new[1:N, 1:N] = u_c + deltat * (epsilon * (u_c - u_c**3 - v_c + F_c) + Du * lap_u)
            v_new[1:N, 1:N] = v_c + deltat * (u_c - alpha * v_c + beta + Dv * lap_v)

            #Paredes sólidas
            """ u_new[0, :] = u_min
            u_new[-1, :] = u_min
            u_new[:, 0] = u_min
            u_new[:, -1] = u_min

            v_new[0, :] = v_min
            v_new[-1, :] = v_min
            v_new[:, 0] = v_min
            v_new[:, -1] = v_min """

            #Guardo u y v
            u = np.copy(u_new)
            v = np.copy(v_new)

            if t % plot_cada == 0:
                imagen.set_data(u)            
                ax.set_title(f'Paso de tiempo: {t}')
                #lt.pause(0.001)    
                ruta_archivo = os.path.join(carpeta_salida, f'frame_{t}.png')
                plt.savefig(ruta_archivo)
                print(f'Calculado t={t} y fotograma guardado.')          
        
        #plt.ioff() # Apagamos el modo interactivo al final
        #plt.show()

    else:
        #carpeta_salida = 'Plots_2'
        carpeta_salida = 'LAB_vuelta'
        os.makedirs(carpeta_salida, exist_ok=True)

        #Bucle
        for t in range(T):

            # Definimos todo menos los bordes, equivalente a range(1,N)
            u_c = u[1:N, 1:N]
            v_c = v[1:N, 1:N]
            F_c = F[1:N, 1:N]

            # Laplacianos vectorizados 
            lap_u = (u[2:N+1, 1:N] + u[0:N-1, 1:N] + u[1:N, 2:N+1] + u[1:N, 0:N-1] - 4*u_c) / deltax**2
            lap_v = (v[2:N+1, 1:N] + v[0:N-1, 1:N] + v[1:N, 2:N+1] + v[1:N, 0:N-1] - 4*v_c) / deltax**2

            # Actualización de las matrices interiores 
            u_new[1:N, 1:N] = u_c + deltat * (epsilon * (u_c - u_c**3 - v_c + F_c) + Du * lap_u)
            v_new[1:N, 1:N] = v_c + deltat * (u_c - alpha * v_c + beta + Dv * lap_v)

            #Paredes sólidas
            """ u_new[0, :] = u_min
            u_new[-1, :] = u_min
            u_new[:, 0] = u_min
            u_new[:, -1] = u_min

            v_new[0, :] = v_min
            v_new[-1, :] = v_min
            v_new[:, 0] = v_min
            v_new[:, -1] = v_min """

            #CONDICIONES DE DIRICHLET PARA LOS EXTREMOS

            u_new[5:45, 5:45] = u_max
            u_new[N-45:N-5, N-45:N-5] = u_max
            v_new[5:45, 5:45] = v_max
            v_new[N-45:N-5, N-45:N-5] = v_max

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
        
        #plt.ioff() # Apagamos el modo interactivo al final
        #plt.show()

        
    
    return u,v


