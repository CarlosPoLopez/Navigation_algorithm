import numpy as np
import time
import os
from bucle import dufort_frankel
from generar_laberinto import generar_laberinto_multiruta, dimx_exacto
from collections import deque


def laberinto_tiene_solucion(matriz, start, end):
    """Quickly check whether a corridor path (zeros) exists from start to end.

    Uses BFS over 4-connectivity, treating 0 as corridor and 1 as wall.
    """
    N_filas, N_cols = matriz.shape
    visitados = set()
    cola = deque([start])
    visitados.add(start)

    # Allowed moves: up, down, left, right
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while cola:
        x, y = cola.popleft()

        if (x, y) == end:
            return True

        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N_filas and 0 <= ny < N_cols:
                if matriz[nx, ny] == 0 and (nx, ny) not in visitados:
                    visitados.add((nx, ny))
                    cola.append((nx, ny))

    return False


# ── Maze geometry ────────────────────────────────────────────────────────────
# canal = cell size (px), gp = wall thickness (px), nc = cells per side.
# dimx_exacto picks the grid size so the cells fit with no leftover border.
canal = 60
gp = 5
nc = 9
dimx = dimx_exacto(nc=nc, canal=canal, grosor_pared=gp)

N = dimx - 1
T = 800000
plot_cada = 20000
fase = 1

# Generate a maze with two disjoint routes (short + long) between the corners.
# The multiruta generator always returns a solvable maze, but we double-check.
matriz_laberinto = generar_laberinto_multiruta(dimx=dimx, canal=canal,
                                               grosor_pared=gp, n_caminos=2)
assert laberinto_tiene_solucion(matriz_laberinto, start=(30, 30), end=(N-30, N-30)), \
    "Generated maze has no solution"

# ── FitzHugh-Nagumo parameters ───────────────────────────────────────────────
deltat = 0.002
deltax = 0.09
epsilon, alpha, beta, Du, Dv, F_pasillo, F_pared = 10.0, 8.0, -0.33, 0.05, 4.0, 0, -3.0
F_matriz = np.where(matriz_laberinto == 1, F_pared, F_pasillo)

u_min, u_max, v_min, v_max = -0.9109, 0.9581, -0.1551, 0.0785

# Initial u and v fields
u_inicial = np.full((N+1, N+1), u_min)
v_inicial = np.full((N+1, N+1), v_min)

# Stable state in the corner: the autowave originates here (maze start/end)
u_inicial[5:55, 5:55] = u_max
v_inicial[5:55, 5:55] = v_max


inicio = time.time()

u_final, v_final = dufort_frankel(u_inicial, v_inicial, u_max, v_max, u_min, v_min,
                                  deltat, deltax, N, T, epsilon, alpha, beta,
                                  Du, Dv, F_matriz, plot_cada, fase)

fin = time.time()
print(f'Simulation finished in {round((fin-inicio)/60, 2)} minutes')

# Save data for phase 2
os.makedirs('Matrices_ida', exist_ok=True)
np.save('Matrices_ida/estado_u_ida.npy', u_final)
np.save('Matrices_ida/estado_v_ida.npy', v_final)
np.save('Matrices_ida/matriz_F_laberinto.npy', F_matriz)
print("Saved! .npy files ready for phase 2.")
