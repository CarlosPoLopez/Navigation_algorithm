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
T = 240000
plot_every = 20000
phase = 2

deltat = 0.002
deltax = 0.25
epsilon, alpha, beta, Du, Dv = 10.0, 5.0, 0.1, 0.6, 5.0

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

# Also save a CNN-ready training sample for Part 2 (Navigation_CNN): the maze field F and
# the solution field u packed into one .npz, with the keys ('maze', 'solution') and the
# sample_<id>.npz naming that its data loader expects. SLURM_ARRAY_TASK_ID, if set, gives
# each run of a batch a unique id.
os.makedirs('dataset', exist_ok=True)
sample_id = os.environ.get('SAMPLE_ID') or os.environ.get('SLURM_ARRAY_TASK_ID', '0')
np.savez_compressed(f'dataset/sample_{sample_id}.npz', maze=F_matrix, solution=u_final)
print(f"Saved! dataset/sample_{sample_id}.npz")
