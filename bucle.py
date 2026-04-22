import numpy as np
import matplotlib.pyplot as plt
import os

def dufort_frankel(u, v, u_max, v_max, u_min, v_min, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F, plot_cada, fase):

    u_new = np.copy(u)
    v_new = np.copy(v)
    
    # Inicializamos el pasado
    u_old = np.copy(u)
    v_old = np.copy(v)

    # Constantes del método de Dufort-Frankel
    Cu = (2.0 * deltat * Du) / (deltax**2)
    Cv = (2.0 * deltat * Dv) / (deltax**2)

    #Configuración plots
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
        carpeta_salida = 'LAB_ida' 
        os.makedirs(carpeta_salida, exist_ok=True)

        #Bucle
        for t in range(T):

            u_old_c = u_old[1:N, 1:N]
            v_old_c = v_old[1:N, 1:N]
            F_c = F[1:N, 1:N]

            # Suma de los vecinos evaluada en el PRESENTE 
            sum_u = u[2:N+1, 1:N] + u[0:N-1, 1:N] + u[1:N, 2:N+1] + u[1:N, 0:N-1]
            sum_v = v[2:N+1, 1:N] + v[0:N-1, 1:N] + v[1:N, 2:N+1] + v[1:N, 0:N-1]

            # Funciones cinéticas evaluadas en el PASADO 
            funu = epsilon * (u_old_c - u_old_c**3 - v_old_c + F_c)
            funv = u_old_c - alpha * v_old_c + beta

            # Ecuación central Dufort-Frankel para calcular el FUTURO 
            u_new[1:N, 1:N] = (2.0 * deltat * funu + u_old_c * (1.0 - 2.0 * Cu) + Cu * sum_u) / (1.0 + 2.0 * Cu)
            v_new[1:N, 1:N] = (2.0 * deltat * funv + v_old_c * (1.0 - 2.0 * Cv) + Cv * sum_v) / (1.0 + 2.0 * Cv)

            #Paredes sólidas
            u_new[0, :] = u_min
            u_new[-1, :] = u_min
            u_new[:, 0] = u_min
            u_new[:, -1] = u_min

            v_new[0, :] = v_min
            v_new[-1, :] = v_min
            v_new[:, 0] = v_min
            v_new[:, -1] = v_min

            # Actualización de matrices en el tiempo
            u_old = np.copy(u)
            v_old = np.copy(v)

            # El nuevo presente es el promedio del futuro y el presente 
            u = (np.copy(u_new) + u) / 2.0
            v = (np.copy(v_new) + v) / 2.0

            if t % plot_cada == 0:
                imagen.set_data(u)            
                ax.set_title(f'Paso de tiempo: {t}')
                ruta_archivo = os.path.join(carpeta_salida, f'frame_{t}.png')
                plt.savefig(ruta_archivo)
                print(f'Calculado t={t} y fotograma guardado.')          

    else:
        carpeta_salida = 'LAB_vuelta'
        os.makedirs(carpeta_salida, exist_ok=True)

        #Bucle
        for t in range(T):

            # Definimos el PASADO (u3, v3 en C)
            u_old_c = u_old[1:N, 1:N]
            v_old_c = v_old[1:N, 1:N]
            F_c = F[1:N, 1:N]

            # Suma de los vecinos evaluada en el PRESENTE (u2, v2 en C)
            sum_u = u[2:N+1, 1:N] + u[0:N-1, 1:N] + u[1:N, 2:N+1] + u[1:N, 0:N-1]
            sum_v = v[2:N+1, 1:N] + v[0:N-1, 1:N] + v[1:N, 2:N+1] + v[1:N, 0:N-1]

            # Funciones cinéticas evaluadas en el PASADO (u3, v3 en C)
            funu = epsilon * (u_old_c - u_old_c**3 - v_old_c + F_c)
            funv = u_old_c - alpha * v_old_c + beta

            # Ecuación central Dufort-Frankel para calcular el FUTURO (u1, v1 en C)
            u_new[1:N, 1:N] = (2.0 * deltat * funu + u_old_c * (1.0 - 2.0 * Cu) + Cu * sum_u) / (1.0 + 2.0 * Cu)
            v_new[1:N, 1:N] = (2.0 * deltat * funv + v_old_c * (1.0 - 2.0 * Cv) + Cv * sum_v) / (1.0 + 2.0 * Cv)

            
            u_new[5:45, 5:45] = u_max
            u_new[N-45:N-5, N-45:N-5] = u_max
            v_new[5:45, 5:45] = v_max
            v_new[N-45:N-5, N-45:N-5] = v_max

            # Actualización de matrices en el tiempo
            u_old = np.copy(u)  
            v_old = np.copy(v)

            # El nuevo presente es el promedio del futuro y el presente (suavizado del C)
            u = (np.copy(u_new) + u) / 2.0
            v = (np.copy(v_new) + v) / 2.0
            
            #CONDICIONES DE DIRICHLET PARA LOS EXTREMOS
            """ u[5:45, 5:45] = u_max
            u[N-45:N-5, N-45:N-5] = u_max
            v[5:45, 5:45] = v_max
            v[N-45:N-5, N-45:N-5] = v_max """

            if t % plot_cada == 0:
                imagen.set_data(u)            
                ax.set_title(f'Paso de tiempo: {t}')
                ruta_archivo = os.path.join(carpeta_salida, f'frame_{t}.png')
                plt.savefig(ruta_archivo)
                print(f'Calculado t={t} y fotograma guardado.')          
    
    return u,v