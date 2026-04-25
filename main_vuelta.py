import numpy as np
import time
import os
from bucle import FCTS

N = 500
T = 180000
plot_cada = 20000 
fase = 2

deltat = 0.002
deltax = 0.25      
epsilon, alpha, beta, Du, Dv = 10.0, 5.0, 0.1, 1, 5.0
F_matriz = np.load('Matrices_ida/matriz_F_laberinto.npy')

""" u_min, u_max, v_min, v_max = -0.6505, 0.7526, -0.3752, 0.3263 """
u_min, u_max, v_min, v_max = -0.6505, 0.877, -0.3752, 0.197

#Matrices u y v al inicio de la simulación
u_inicial = np.load('Matrices_ida/estado_u_ida.npy')
v_inicial = np.load('Matrices_ida/estado_v_ida.npy')


inicio = time.time()

u_final , v_final = FCTS(u_inicial, v_inicial, u_max, v_max, u_min, v_min, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F_matriz, plot_cada, fase)

fin = time.time()
print(f'Simulación terminada en {round((fin-inicio)/60, 2)}minutos')

#Guardar datos
os.makedirs('Matrices_vuelta', exist_ok=True)
np.save('Matrices_vuelta/estado_u_vuelta.npy', u_final)  
np.save('Matrices_vuelta/estado_v_vuelta.npy', v_final)
np.save('Matrices_vuelta/matriz_F_laberinto.npy', F_matriz)
print("¡Guardado! Archivos .npy generados.") 