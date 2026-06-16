import numpy as np
import time
import os
from solver import dufort_frankel
from maze import generate_maze_multiroute, exact_dimx
from collections import deque


def maze_has_solution(matrix, start, end):
    """Quickly check whether a corridor path (zeros) exists from start to end

    Uses BFS over 4-connectivity, treating 0 as corridor and 1 as wall
    """
    n_rows, n_cols = matrix.shape
    visited = set()
    queue = deque([start])
    visited.add(start)

    # Allowed moves: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()

        if (x, y) == end:
            return True

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n_rows and 0 <= ny < n_cols:
                if matrix[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False


# ── Maze geometry ────────────────────────────────────────────────────────────
# cell_size = cell size (px), wall_thickness = wall thickness (px),
# n_cells = cells per side. exact_dimx picks the grid size so the cells fit
# with no leftover border.
cell_size = 60
wall_thickness = 5
n_cells = 9
dimx = exact_dimx(n_cells=n_cells, cell_size=cell_size, wall_thickness=wall_thickness)

N = dimx - 1
T = 800000
plot_every = 20000
phase = 1

# Generate a maze with two disjoint routes (short + long) between the corners.
# The multiroute generator always returns a solvable maze, but we double-check.
maze = generate_maze_multiroute(dimx=dimx, cell_size=cell_size,
                                wall_thickness=wall_thickness, n_paths=2)
assert maze_has_solution(maze, start=(30, 30), end=(N-30, N-30)), \
    "Generated maze has no solution"

# ── FitzHugh-Nagumo parameters ───────────────────────────────────────────────
deltat = 0.002
deltax = 0.09
epsilon, alpha, beta, Du, Dv, F_corridor, F_wall = 10.0, 8.0, -0.33, 0.05, 4.0, 0, -3.0
F_matrix = np.where(maze == 1, F_wall, F_corridor)

u_min, u_max, v_min, v_max = -0.9109, 0.9581, -0.1551, 0.0785

# Initial u and v fields
u_init = np.full((N+1, N+1), u_min)
v_init = np.full((N+1, N+1), v_min)

# Stable state in the corner: the autowave originates here (maze start/end)
u_init[5:55, 5:55] = u_max
v_init[5:55, 5:55] = v_max


start_time = time.time()

u_final, v_final = dufort_frankel(u_init, v_init, u_max, v_max, u_min, v_min,
                                  deltat, deltax, N, T, epsilon, alpha, beta,
                                  Du, Dv, F_matrix, plot_every, phase)

end_time = time.time()
print(f'Simulation finished in {round((end_time-start_time)/60, 2)} minutes')

# Save data for phase 2
os.makedirs('matrices_expansion', exist_ok=True)
np.save('matrices_expansion/state_u.npy', u_final)
np.save('matrices_expansion/state_v.npy', v_final)
np.save('matrices_expansion/maze_F.npy', F_matrix)
print("Saved! .npy files ready for phase 2.")
