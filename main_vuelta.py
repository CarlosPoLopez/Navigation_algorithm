import numpy as np
import time
import os
from bucle import FTCS

# Load the phase-1 final state
F_matriz = np.load('Matrices_ida/matriz_F_laberinto.npy')
u_inicial = np.load('Matrices_ida/estado_u_ida.npy')
v_inicial = np.load('Matrices_ida/estado_v_ida.npy')

# Derive the grid size from the loaded fields so it matches phase 1
N = F_matriz.shape[0] - 1
T = 180000
plot_cada = 20000
fase = 2

deltat = 0.002
deltax = 0.25
epsilon, alpha, beta, Du, Dv = 10.0, 5.0, 0.1, 1, 5.0

u_min, u_max, v_min, v_max = -0.6505, 0.877, -0.3752, 0.197


inicio = time.time()

u_final, v_final = FTCS(u_inicial, v_inicial, u_max, v_max, u_min, v_min,
                        deltat, deltax, N, T, epsilon, alpha, beta,
                        Du, Dv, F_matriz, plot_cada, fase)

fin = time.time()
print(f'Simulation finished in {round((fin-inicio)/60, 2)} minutes')

# Save data
os.makedirs('Matrices_vuelta', exist_ok=True)
np.save('Matrices_vuelta/estado_u_vuelta.npy', u_final)
np.save('Matrices_vuelta/estado_v_vuelta.npy', v_final)
np.save('Matrices_vuelta/matriz_F_laberinto.npy', F_matriz)
print("Saved! .npy files generated.")
