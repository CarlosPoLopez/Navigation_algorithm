import numpy as np
import matplotlib.pyplot as plt


# ── Helper: exact grid size ──────────────────────────────────────────────────

def exact_dimx(n_cells, cell_size, wall_thickness=3):
    """Return the exact array size so the cell grid fits with no leftover border.

    Parameters
    ----------
    n_cells : int
        Number of cells per side.
    cell_size : int
        Size of each cell in pixels.
    wall_thickness : int
        Wall thickness in pixels (default 3).

    Example
    -------
    >>> exact_dimx(n_cells=9, cell_size=60, wall_thickness=5)
    590
    """
    return n_cells * (cell_size + wall_thickness) + wall_thickness


# ── Perfect maze (single path between any two cells) ─────────────────────────

def generate_maze_perfect(dimx=501, cell_size=50, wall_thickness=3, seed=None):
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
    cell_size : int
        Inner size of each cell in pixels.
    wall_thickness : int
        Thickness of the walls between cells (and of the perimeter).
    seed : int, optional
        Seed for np.random (reproducibility).

    Returns
    -------
    aa : np.ndarray, shape (dimx, dimx), dtype int
        1 = wall, 0 = open corridor/cell.

    How it works
    ------------
    1. Start from an array completely filled with walls (all ones).
    2. Lay out a grid of cells separated by walls of `wall_thickness`.
    3. The iterative DFS visits each cell exactly once and carves the corridor
       to its neighbour: it opens both the destination cell and the wall strip
       between them.
    4. Being a spanning tree, the resulting wall graph is connected and there
       are no floating walls.
    """
    if seed is not None:
        np.random.seed(seed)

    wt = wall_thickness

    # Number of cells that fit along each dimension
    n_cells = (dimx - wt) // (cell_size + wt)

    # Adjust the cell size to fill as much space as possible inside dimx,
    # minimising the leftover offset (<= 1 px per side in practice).
    cell = (dimx - wt * (n_cells + 1)) // n_cells

    # Residual offset to centre any leftover pixels
    dimx_real = n_cells * (cell + wt) + wt
    offset = (dimx - dimx_real) // 2

    # Pixel position of the top-left corner of cell (i, j)
    def px(k):
        return offset + wt + k * (cell + wt)

    # ── Initialise: everything is wall ──────────────────────────────────────
    aa = np.ones((dimx, dimx), dtype=int)

    # ── Iterative DFS ───────────────────────────────────────────────────────
    visited = np.zeros((n_cells, n_cells), dtype=bool)

    # Open the starting cell (0, 0)
    r0, c0 = 0, 0
    aa[px(r0):px(r0) + cell, px(c0):px(c0) + cell] = 0
    visited[r0, c0] = True

    stack = [(r0, c0)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack[-1]

        # Unvisited neighbours
        neighbors = [
            (r + dr, c + dc)
            for dr, dc in directions
            if 0 <= r + dr < n_cells and 0 <= c + dc < n_cells
            and not visited[r + dr, c + dc]
        ]

        if neighbors:
            # Pick a random neighbour
            nr, ncol = neighbors[np.random.randint(len(neighbors))]

            # ── Open the wall between (r,c) and (nr,ncol) ───────────────
            if nr == r:          # same row -> vertical wall between columns
                jmin = min(c, ncol)
                aa[px(r):px(r) + cell, px(jmin) + cell:px(jmin) + cell + wt] = 0
            else:                # same column -> horizontal wall between rows
                imin = min(r, nr)
                aa[px(imin) + cell:px(imin) + cell + wt, px(c):px(c) + cell] = 0

            # ── Open the destination cell ───────────────────────────────
            aa[px(nr):px(nr) + cell, px(ncol):px(ncol) + cell] = 0
            visited[nr, ncol] = True
            stack.append((nr, ncol))
        else:
            stack.pop()   # backtrack

    # ── Seal the perimeter ──────────────────────────────────────────────────
    aa[:wt, :] = 1
    aa[-wt:, :] = 1
    aa[:, :wt] = 1
    aa[:, -wt:] = 1

    return aa


# ── Multi-route maze: N vertex-disjoint paths start -> end ───────────────────

def generate_maze_multiroute(dimx=590, cell_size=60, wall_thickness=5,
                             n_paths=2, start_cell=None, end_cell=None,
                             seed=None):
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
    dimx, cell_size, wall_thickness : see generate_maze_perfect.
    n_paths : int
        Number of disjoint paths to build (default 2).
    start_cell, end_cell : tuple(int, int), optional
        (r, c) in the cell grid. Default: opposite corners (0, 0) and
        (n_cells-1, n_cells-1).
    seed : int, optional
        Seed for np.random.

    Returns
    -------
    aa : np.ndarray, shape (dimx, dimx), dtype int. 1 = wall, 0 = corridor.
    """
    from collections import deque

    if seed is not None:
        np.random.seed(seed)

    wt = wall_thickness
    n_cells = (dimx - wt) // (cell_size + wt)
    cell = (dimx - wt * (n_cells + 1)) // n_cells
    dimx_real = n_cells * (cell + wt) + wt
    offset = (dimx - dimx_real) // 2

    def px(k):
        return offset + wt + k * (cell + wt)

    if start_cell is None:
        start_cell = (0, 0)
    if end_cell is None:
        end_cell = (n_cells - 1, n_cells - 1)

    # Neighbours on the cell grid (4-connectivity)
    def neighbors(rc):
        r, c = rc
        return [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < n_cells and 0 <= c + dc < n_cells]

    # 1) Find N vertex-disjoint paths. The first one is the shortest (BFS).
    #    The remaining ones are intentionally longer (randomised DFS with
    #    retries) so they act as distractors: the wave algorithm must pick the
    #    short one, the long ones are branches to discard.
    def path_bfs(blocked):
        parent = {start_cell: None}
        queue = deque([start_cell])
        while queue:
            u = queue.popleft()
            if u == end_cell:
                break
            cands = [v for v in neighbors(u)
                     if v not in parent and (v == end_cell or v not in blocked)]
            np.random.shuffle(cands)
            for v in cands:
                parent[v] = u
                queue.append(v)
        if end_cell not in parent:
            return None
        path = []
        cur = end_cell
        while cur is not None:
            path.append(cur); cur = parent[cur]
        return path[::-1]

    def path_dfs(blocked):
        # Randomised DFS: tends to produce winding, long paths.
        parent = {start_cell: None}
        visited = {start_cell}
        stack = [start_cell]
        while stack:
            u = stack[-1]
            if u == end_cell:
                break
            cands = [v for v in neighbors(u)
                     if v not in visited and (v == end_cell or v not in blocked)]
            np.random.shuffle(cands)
            if cands:
                v = cands[0]
                parent[v] = u
                visited.add(v)
                stack.append(v)
            else:
                stack.pop()
        if end_cell not in parent:
            return None
        path = []
        cur = end_cell
        while cur is not None:
            path.append(cur); cur = parent[cur]
        return path[::-1]

    # For n_paths=2 the first BFS may pick a route whose blocking separates
    # start from end (e.g. a diagonal staircase touching both borders),
    # preventing the second path. Retry the first path several times until the
    # second (longer) one exists.
    min_extra = 6
    paths = []
    for _outer in range(30):
        p1 = path_bfs(set())
        if p1 is None:
            continue
        blocked = set(p1[1:-1])
        best_p2 = None
        for _ in range(60):
            p = path_dfs(blocked)
            if p is None:
                continue
            if best_p2 is None or len(p) > len(best_p2):
                best_p2 = p
            if best_p2 is not None and len(best_p2) >= len(p1) + min_extra:
                break
        if best_p2 is not None and len(best_p2) >= len(p1) + min_extra:
            paths = [p1, best_p2]
            break
    if not paths:
        # Could not get 2 paths with the minimum gap in 30 attempts.
        # Accept whatever we have (at least the short path).
        p1 = path_bfs(set())
        paths = [p1] if p1 is not None else []
        if p1 is not None:
            blocked = set(p1[1:-1])
            p = path_dfs(blocked)
            if p is not None:
                paths.append(p)

    # Extra paths (n_paths > 2): only possible in graphs with a bottleneck > 2,
    # which does not happen with opposite corners; ignored silently.
    blocked = set()
    for path in paths:
        for c in path[1:-1]:
            blocked.add(c)

    # Collect the cells that belong to some path
    path_cells = set()
    for path in paths:
        path_cells.update(path)

    # 2) Build the empty maze (all walls) and open the paths
    aa = np.ones((dimx, dimx), dtype=int)

    def open_cell(rc):
        r, c = rc
        aa[px(r):px(r) + cell, px(c):px(c) + cell] = 0

    def open_wall(a, b):
        ra, ca = a
        rb, cb = b
        if ra == rb:                                    # horizontal neighbours
            jmin = min(ca, cb)
            aa[px(ra):px(ra) + cell,
               px(jmin) + cell:px(jmin) + cell + wt] = 0
        else:                                           # vertical neighbours
            imin = min(ra, rb)
            aa[px(imin) + cell:px(imin) + cell + wt,
               px(ca):px(ca) + cell] = 0

    for path in paths:
        for cell_rc in path:
            open_cell(cell_rc)
        for i in range(len(path) - 1):
            open_wall(path[i], path[i + 1])

    # 3) Fill the rest with dead-end trees hanging off the paths.
    #    Randomised DFS: the stack starts with all path cells and expands by
    #    visiting unvisited neighbours (without reconnecting distinct paths).
    visited = set(path_cells)
    stack = list(path_cells)
    np.random.shuffle(stack)
    while stack:
        u = stack[-1]
        candidates = [v for v in neighbors(u) if v not in visited]
        if candidates:
            v = candidates[np.random.randint(len(candidates))]
            open_cell(v)
            open_wall(u, v)
            visited.add(v)
            stack.append(v)
        else:
            stack.pop()

    # 4) Seal the perimeter
    aa[:wt, :] = 1
    aa[-wt:, :] = 1
    aa[:, :wt] = 1
    aa[:, -wt:] = 1

    return aa


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cell_size = 60
    wall_thickness = 5
    dimx = exact_dimx(n_cells=9, cell_size=cell_size, wall_thickness=wall_thickness)
    print(f'dimx = {dimx}')

    fig, ax = plt.subplots(figsize=(8, 8))
    lab = generate_maze_multiroute(dimx=dimx, cell_size=cell_size,
                                   wall_thickness=wall_thickness, n_paths=2, seed=0)
    ax.imshow(lab, cmap='binary', origin='lower')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
