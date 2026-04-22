import numpy as np
import time
import os
from bucle import dufort_frankel
from generar_laberinto import generar_laberinto_3


N = 500
L = N + 1
T = 900000
plot_cada = 20000
fase = 1
matriz_laberinto = generar_laberinto_3()

deltat = 0.002 
deltax = 0.09
epsilon, alpha, beta, Du, Dv, F_pasillo, F_pared = 10.0, 8.0, -0.33, 0.05, 2.0, 0, -3.0
F_matriz = np.where(matriz_laberinto == 1, F_pared, F_pasillo)

u_min, u_max, v_min, v_max = -0.9109, 0.9581, -0.1551, 0.0785   

#Matrices u y v al inicio de la simulación
u_inicial = np.full((N+1, N+1), u_min)
v_inicial = np.full((N+1,N+1), v_min)


#Estado estable en las esquinas (de ahí surgen las autoondas, inicio y final del laberinto)
u_inicial[5:45, 5:45] = u_max
v_inicial[5:45, 5:45] = v_max


inicio = time.time()

u_final , v_final = dufort_frankel(u_inicial, v_inicial, u_max, v_max, u_min, v_min, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F_matriz, plot_cada, fase)

fin = time.time()
print(f'Simulación terminada en {round((fin-inicio)/60, 2)}minutos')

#Guardar datos
os.makedirs('Matrices_ida', exist_ok=True)
np.save('Matrices_ida/estado_u_ida.npy', u_final)  
np.save('Matrices_ida/estado_v_ida.npy', v_final)
np.save('Matrices_ida/matriz_F_laberinto.npy', F_matriz)
print("¡Guardado! Archivos .npy generados listos para la Fase 2.")