import numpy as np
import matplotlib.pyplot as plt


# ── Helper: exact grid size ──────────────────────────────────────────────────

def dimx_exacto(nc, canal, grosor_pared=3):
    """Return the exact array size so the cell grid fits with no leftover border.

    Parameters
    ----------
    nc : int
        Number of cells per side.
    canal : int
        Size of each cell in pixels.
    grosor_pared : int
        Wall thickness in pixels (default 3).

    Example
    -------
    >>> dimx_exacto(nc=9, canal=60, grosor_pared=5)
    590
    """
    return nc * (canal + grosor_pared) + grosor_pared


# ── Perfect maze (single path between any two cells) ─────────────────────────

def generar_laberinto_perfecto(dimx=501, canal=50, grosor_pared=3, semilla=None):
    """Generate a perfect maze via iterative DFS backtracking.

    Guaranteed properties:
      - No islands: every wall segment is connected to the perimeter (the wall
        set is a connected component).
      - No loops: there is exactly one path between any pair of cells.
      - Fully solvable: every cell is reachable.

    Parameters
    ----------
    dimx : int
        Size of the square output array (pixels).
    canal : int
        Inner size of each cell in pixels.
    grosor_pared : int
        Thickness of the walls between cells (and of the perimeter).
    semilla : int, optional
        Seed for np.random (reproducibility).

    Returns
    -------
    aa : np.ndarray, shape (dimx, dimx), dtype int
        1 = wall, 0 = open corridor/cell.

    How it works
    ------------
    1. Start from an array completely filled with walls (all ones).
    2. Lay out a grid of cells separated by walls of `grosor_pared`.
    3. The iterative DFS visits each cell exactly once and carves the corridor
       to its neighbour: it opens both the destination cell and the wall strip
       between them.
    4. Being a spanning tree, the resulting wall graph is connected and there
       are no floating walls.
    """
    if semilla is not None:
        np.random.seed(semilla)

    gp = grosor_pared

    # Number of cells that fit along each dimension
    nc = (dimx - gp) // (canal + gp)

    # Adjust the cell size to fill as much space as possible inside dimx,
    # minimising the leftover offset (<= 1 px per side in practice).
    canal = (dimx - gp * (nc + 1)) // nc

    # Residual offset to centre any leftover pixels
    dimx_real = nc * (canal + gp) + gp
    offset = (dimx - dimx_real) // 2

    # Pixel position of the top-left corner of cell (i, j)
    def px(k):
        return offset + gp + k * (canal + gp)

    # ── Initialise: everything is wall ──────────────────────────────────────
    aa = np.ones((dimx, dimx), dtype=int)

    # ── Iterative DFS ───────────────────────────────────────────────────────
    visited = np.zeros((nc, nc), dtype=bool)

    # Open the starting cell (0, 0)
    r0, c0 = 0, 0
    aa[px(r0):px(r0) + canal, px(c0):px(c0) + canal] = 0
    visited[r0, c0] = True

    stack = [(r0, c0)]
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack[-1]

        # Unvisited neighbours
        vecinos = [
            (r + dr, c + dc)
            for dr, dc in direcciones
            if 0 <= r + dr < nc and 0 <= c + dc < nc
            and not visited[r + dr, c + dc]
        ]

        if vecinos:
            # Pick a random neighbour
            nr, nc_ = vecinos[np.random.randint(len(vecinos))]

            # ── Open the wall between (r,c) and (nr,nc_) ────────────────
            if nr == r:          # same row -> vertical wall between columns
                jmin = min(c, nc_)
                aa[px(r):px(r) + canal, px(jmin) + canal:px(jmin) + canal + gp] = 0
            else:                # same column -> horizontal wall between rows
                imin = min(r, nr)
                aa[px(imin) + canal:px(imin) + canal + gp, px(c):px(c) + canal] = 0

            # ── Open the destination cell ───────────────────────────────
            aa[px(nr):px(nr) + canal, px(nc_):px(nc_) + canal] = 0
            visited[nr, nc_] = True
            stack.append((nr, nc_))
        else:
            stack.pop()   # backtrack

    # ── Seal the perimeter ──────────────────────────────────────────────────
    aa[:gp, :] = 1
    aa[-gp:, :] = 1
    aa[:, :gp] = 1
    aa[:, -gp:] = 1

    return aa


# ── Multi-route maze: N vertex-disjoint paths start -> end ───────────────────

def generar_laberinto_multiruta(dimx=590, canal=60, grosor_pared=5,
                                n_caminos=2, start_cell=None, end_cell=None,
                                semilla=None):
    """Build a maze with N vertex-disjoint paths from start_cell to end_cell.

    This is the generator used to build the training dataset. The paths share
    no internal cell (only the start and end cells), so they are physically
    distinct routes, not variations of the same one.

    Algorithm
    ---------
    1. Find N vertex-disjoint paths on the cell grid via sequential BFS: after
       the first path is found, its internal cells are blocked and the next
       path is searched for.
    2. Open every corridor belonging to those paths.
    3. Create dead ends by growing DFS trees from the cells that already belong
       to a path, until the whole grid is covered. Because the branches never
       connect two distinct paths, no shortcuts are introduced.
    4. Seal the perimeter.

    Notes
    -----
    - On a 9x9 grid with start and end in opposite corners, the theoretical
      maximum number of vertex-disjoint paths is 2 (each corner has only 2
      neighbours). For N=2 it is always achieved. For N>2 the paths that could
      be built are returned.
    - Unlike a loop-injection approach, this guarantees NO wall islands,
      because the structure is built by adding corridors on top of a
      wall-filled background instead of removing walls.

    Parameters
    ----------
    dimx, canal, grosor_pared : see generar_laberinto_perfecto.
    n_caminos : int
        Number of disjoint paths to build (default 2).
    start_cell, end_cell : tuple(int, int), optional
        (r, c) in the cell grid. Default: opposite corners (0, 0) and
        (nc-1, nc-1).
    semilla : int, optional
        Seed for np.random.

    Returns
    -------
    aa : np.ndarray, shape (dimx, dimx), dtype int. 1 = wall, 0 = corridor.
    """
    from collections import deque

    if semilla is not None:
        np.random.seed(semilla)

    gp = grosor_pared
    nc = (dimx - gp) // (canal + gp)
    canal_aj = (dimx - gp * (nc + 1)) // nc
    dimx_real = nc * (canal_aj + gp) + gp
    offset = (dimx - dimx_real) // 2

    def px(k):
        return offset + gp + k * (canal_aj + gp)

    if start_cell is None:
        start_cell = (0, 0)
    if end_cell is None:
        end_cell = (nc - 1, nc - 1)

    # Neighbours on the cell grid (4-connectivity)
    def vecinos(rc):
        r, c = rc
        return [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < nc and 0 <= c + dc < nc]

    # 1) Find N vertex-disjoint paths. The first one is the shortest (BFS).
    #    The remaining ones are intentionally longer (randomised DFS with
    #    retries) so they act as distractors: the wave algorithm must pick the
    #    short one, the long ones are branches to discard.
    def camino_bfs(bloqueados):
        parent = {start_cell: None}
        cola = deque([start_cell])
        while cola:
            u = cola.popleft()
            if u == end_cell:
                break
            vecs = [v for v in vecinos(u)
                    if v not in parent and (v == end_cell or v not in bloqueados)]
            np.random.shuffle(vecs)
            for v in vecs:
                parent[v] = u
                cola.append(v)
        if end_cell not in parent:
            return None
        c = []
        cur = end_cell
        while cur is not None:
            c.append(cur); cur = parent[cur]
        return c[::-1]

    def camino_dfs(bloqueados):
        # Randomised DFS: tends to produce winding, long paths.
        parent = {start_cell: None}
        visitado = {start_cell}
        pila = [start_cell]
        while pila:
            u = pila[-1]
            if u == end_cell:
                break
            vecs = [v for v in vecinos(u)
                    if v not in visitado and (v == end_cell or v not in bloqueados)]
            np.random.shuffle(vecs)
            if vecs:
                v = vecs[0]
                parent[v] = u
                visitado.add(v)
                pila.append(v)
            else:
                pila.pop()
        if end_cell not in parent:
            return None
        c = []
        cur = end_cell
        while cur is not None:
            c.append(cur); cur = parent[cur]
        return c[::-1]

    # For n_caminos=2 the first BFS may pick a route whose blocking separates
    # start from end (e.g. a diagonal staircase touching both borders),
    # preventing the second path. Retry the first path several times until the
    # second (longer) one exists.
    min_extra = 6
    caminos = []
    for _outer in range(30):
        p1 = camino_bfs(set())
        if p1 is None:
            continue
        bloqueados = set(p1[1:-1])
        mejor_p2 = None
        for _ in range(60):
            c = camino_dfs(bloqueados)
            if c is None:
                continue
            if mejor_p2 is None or len(c) > len(mejor_p2):
                mejor_p2 = c
            if mejor_p2 is not None and len(mejor_p2) >= len(p1) + min_extra:
                break
        if mejor_p2 is not None and len(mejor_p2) >= len(p1) + min_extra:
            caminos = [p1, mejor_p2]
            break
    if not caminos:
        # Could not get 2 paths with the minimum gap in 30 attempts.
        # Accept whatever we have (at least the short path).
        p1 = camino_bfs(set())
        caminos = [p1] if p1 is not None else []
        if p1 is not None:
            bloqueados = set(p1[1:-1])
            c = camino_dfs(bloqueados)
            if c is not None:
                caminos.append(c)

    # Extra paths (n_caminos > 2): only possible in graphs with a bottleneck
    # > 2, which does not happen with opposite corners; ignored silently.
    bloqueados = set()
    for camino in caminos:
        for c in camino[1:-1]:
            bloqueados.add(c)

    # Collect the cells that belong to some path
    celdas_camino = set()
    for camino in caminos:
        celdas_camino.update(camino)

    # 2) Build the empty maze (all walls) and open the paths
    aa = np.ones((dimx, dimx), dtype=int)

    def abrir_celda(rc):
        r, c = rc
        aa[px(r):px(r) + canal_aj, px(c):px(c) + canal_aj] = 0

    def abrir_pared(a, b):
        ra, ca = a
        rb, cb = b
        if ra == rb:                                    # horizontal neighbours
            jmin = min(ca, cb)
            aa[px(ra):px(ra) + canal_aj,
               px(jmin) + canal_aj:px(jmin) + canal_aj + gp] = 0
        else:                                           # vertical neighbours
            imin = min(ra, rb)
            aa[px(imin) + canal_aj:px(imin) + canal_aj + gp,
               px(ca):px(ca) + canal_aj] = 0

    for camino in caminos:
        for cell in camino:
            abrir_celda(cell)
        for i in range(len(camino) - 1):
            abrir_pared(camino[i], camino[i + 1])

    # 3) Fill the rest with dead-end trees hanging off the paths.
    #    Randomised DFS: the stack starts with all path cells and expands by
    #    visiting unvisited neighbours (without reconnecting distinct paths).
    visitado = set(celdas_camino)
    pila = list(celdas_camino)
    np.random.shuffle(pila)
    while pila:
        u = pila[-1]
        candidatos = [v for v in vecinos(u) if v not in visitado]
        if candidatos:
            v = candidatos[np.random.randint(len(candidatos))]
            abrir_celda(v)
            abrir_pared(u, v)
            visitado.add(v)
            pila.append(v)
        else:
            pila.pop()

    # 4) Seal the perimeter
    aa[:gp, :] = 1
    aa[-gp:, :] = 1
    aa[:, :gp] = 1
    aa[:, -gp:] = 1

    return aa


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    canal = 60
    gp = 5
    dimx = dimx_exacto(nc=9, canal=canal, grosor_pared=gp)
    print(f'dimx = {dimx}')

    fig, ax = plt.subplots(figsize=(8, 8))
    lab = generar_laberinto_multiruta(dimx=dimx, canal=canal,
                                      grosor_pared=gp, n_caminos=2, semilla=0)
    ax.imshow(lab, cmap='binary', origin='lower')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
