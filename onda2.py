import numpy as np
import time
import os
from FCTS import FCTS

N = 400
T = 700000
plot_cada = 20000 
fase = 2

deltat = 0.002
deltax = 0.25      
epsilon, alpha, beta, Du, Dv, F = 10.0, 5.0, 0.1, 0.3, 5.0, 0

u_min, u_max, v_min, v_max = -0.6505, 0.7526, -0.3752, 0.3263

#Matrices u y v al inicio de la simulación
u_inicial = np.load('Datos_uv/estado_u_fase1.npy')
v_inicial = np.load('Datos_uv/estado_v_fase1.npy')


inicio = time.time()

u_final , v_final = FCTS(u_inicial, v_inicial, u_max, v_max, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F, plot_cada, fase)

fin = time.time()
print(f'Simulación terminada en {round((fin-inicio)/60, 2)}minutos')

#Guardar datos
os.makedirs('Datos_uv', exist_ok=True)
np.save('Datos_uv/estado_u_fase2.npy', u_final)  
np.save('Datos_uv/estado_v_fase2.npy', v_final)
print("¡Guardado! Archivos .npy generados.") 