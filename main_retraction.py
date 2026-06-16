import numpy as np
import time
import os
from solver import FTCS

# Load the phase 1 final state
F_matrix = np.load('matrices_expansion/maze_F.npy')
u_init = np.load('matrices_expansion/state_u.npy')
v_init = np.load('matrices_expansion/state_v.npy')

# Derive the grid size from the loaded fields so it matches phase 1
N = F_matrix.shape[0] - 1
T = 180000
plot_every = 20000
phase = 2

deltat = 0.002
deltax = 0.25
epsilon, alpha, beta, Du, Dv = 10.0, 5.0, 0.1, 1, 5.0

u_min, u_max, v_min, v_max = -0.6505, 0.877, -0.3752, 0.197


start_time = time.time()

u_final, v_final = FTCS(u_init, v_init, u_max, v_max, u_min, v_min,
                        deltat, deltax, N, T, epsilon, alpha, beta,
                        Du, Dv, F_matrix, plot_every, phase)

end_time = time.time()
print(f'Simulation finished in {round((end_time-start_time)/60, 2)} minutes')

# Save data
os.makedirs('matrices_retraction', exist_ok=True)
np.save('matrices_retraction/state_u.npy', u_final)
np.save('matrices_retraction/state_v.npy', v_final)
np.save('matrices_retraction/maze_F.npy', F_matrix)
print("Saved! .npy files generated.")
