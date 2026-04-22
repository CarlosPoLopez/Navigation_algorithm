import numpy as np
import time
import os
from FCTS import FCTS

N = 500
T = 1000000
plot_cada = 20000
fase = 1

deltat = 0.001
deltax = 0.09 
epsilon, alpha, beta, Du, Dv, F= 10.0, 8.0, -0.33, 0.05, 2.0, 0
F_matriz = np.full((N+1, N+1), F)

u_min, u_max, v_min, v_max = -0.9109, 0.9581, -0.1551, 0.0785   

#Matrices u y v al inicio de la simulación
u_inicial = np.full((N+1, N+1), u_min)
v_inicial = np.full((N+1,N+1), v_min)

#Estado estable en las esquinas (de ahí surgen las autoondas, inicio y final del laberinto)
u_inicial[5:45, 5:45] = u_max
u_inicial[N-45:N-5, N-45:N-5] = u_max
v_inicial[5:45, 5:45] = v_max
v_inicial[N-45:N-5, N-45:N-5] = v_max


inicio = time.time()

u_final , v_final = FCTS(u_inicial, v_inicial, u_max, v_max, u_min, v_min, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F_matriz, plot_cada, fase)

fin = time.time()
print(f'Simulación terminada en {round((fin-inicio)/60, 2)}minutos')

#Guardar datos
os.makedirs('Datos_uv', exist_ok=True)
np.save('Datos_uv/estado_u_fase1.npy', u_final)  
np.save('Datos_uv/estado_v_fase1.npy', v_final)
print("¡Guardado! Archivos .npy generados listos para la Fase 2.")