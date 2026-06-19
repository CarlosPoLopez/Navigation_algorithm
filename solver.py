import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from numba import njit, prange


# ── Numba kernels (one parallel pass per cell, no temporary arrays) ─────────

@njit(parallel=True, cache=True)
def _step_dufort_frankel(u, v, u_old, v_old, u_new, v_new,
                         F, Cu, Cv, deltat, epsilon, alpha, beta, N):
    for i in prange(1, N):
        for j in range(1, N):
            su = u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]
            sv = v[i+1, j] + v[i-1, j] + v[i, j+1] + v[i, j-1]
            react_u = epsilon * (u_old[i, j] - u_old[i, j]**3 - v_old[i, j] + F[i, j])
            react_v = u_old[i, j] - alpha * v_old[i, j] + beta
            u_new[i, j] = (2.0*deltat*react_u + u_old[i, j]*(1.0 - 2.0*Cu) + Cu*su) / (1.0 + 2.0*Cu)
            v_new[i, j] = (2.0*deltat*react_v + v_old[i, j]*(1.0 - 2.0*Cv) + Cv*sv) / (1.0 + 2.0*Cv)
    return u_new, v_new


@njit(parallel=True, cache=True)
def _step_ftcs(u, v, u_new, v_new, F,
               deltat, inv_dx2, epsilon, alpha, beta, Du, Dv, N):
    for i in prange(1, N):
        for j in range(1, N):
            lap_u = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0*u[i, j]) * inv_dx2
            lap_v = (v[i+1, j] + v[i-1, j] + v[i, j+1] + v[i, j-1] - 4.0*v[i, j]) * inv_dx2
            u_new[i, j] = u[i, j] + deltat * (epsilon*(u[i, j] - u[i, j]**3 - v[i, j] + F[i, j]) + Du*lap_u)
            v_new[i, j] = v[i, j] + deltat * (u[i, j] - alpha*v[i, j] + beta + Dv*lap_v)
    return u_new, v_new


def _save_frame(u, F, N, output_dir, t):
    """Save a single frame: the activator field with the maze walls overlaid."""
    os.makedirs(output_dir, exist_ok=True)
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(u, cmap='magma', vmin=-1.5, vmax=1.5, origin='lower')
    overlay = np.zeros((N + 1, N + 1, 4))
    overlay[F < 0] = [0.8, 0.8, 0.8, 0.8]
    ax.imshow(overlay, origin='lower')
    ax.set_title(f'Time step: {t}')
    fig.savefig(os.path.join(output_dir, f'frame_{t}.png'))
    plt.close(fig)


def dufort_frankel(u, v, u_max, v_max, u_min, v_min, deltat, deltax, N, T,
                   epsilon, alpha, beta, Du, Dv, F, plot_every, phase,
                   stop_at_exit=True, exit_threshold=0.0, check_every=200,
                   exit_margin=0):
    """Integrate the FitzHugh-Nagumo system with the Dufort-Frankel scheme.

    Used in phase 1 (expansion): an autowave is launched from the start corner
    and floods the maze along the corridors. Dufort-Frankel is an explicit,
    unconditionally stable leapfrog scheme, which lets the front propagate over
    the long times the expansion needs. The per-step update runs as a parallel
    Numba kernel (one pass per cell, no temporary arrays).

    Parameters
    ----------
    u, v : np.ndarray
        Initial activator (u) and inhibitor (v) fields, shape (N+1, N+1).
    u_max, v_max, u_min, v_min : float
        Stable-state values used to seed/clamp the wave sources.
    deltat, deltax : float
        Time step and spatial step.
    N : int
        Grid index bound (the field is (N+1) x (N+1)).
    T : int
        Number of time steps.
    epsilon, alpha, beta, Du, Dv : float
        FitzHugh-Nagumo kinetic and diffusion parameters.
    F : np.ndarray
        Forcing matrix encoding the maze geometry (negative on walls).
    plot_every : int
        Save a frame every `plot_every` steps (0 disables frames).
    phase : int
        Phase tag (1 = expansion); kept for symmetry with FTCS.
    stop_at_exit : bool
        If True, stop the integration early once the wave front reaches the
        exit corner instead of always running the full T steps. Useful in
        phase 1, where the maze is already flooded long before T is reached.
    exit_threshold : float
        Activator level above which the exit corner is considered reached
        (the resting state is u_min < 0, so 0.0 is a safe default).
    check_every : int
        How often, in steps, to check whether the front has reached the exit.
    exit_margin : int
        Extra steps to keep integrating after the front is first detected at
        the exit, before actually stopping.

    Returns
    -------
    u, v : np.ndarray
        Final activator and inhibitor fields.
    """
    u = np.ascontiguousarray(u, dtype=np.float64)
    v = np.ascontiguousarray(v, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.float64)

    u_old = u.copy()
    v_old = v.copy()
    u_new = u.copy()
    v_new = v.copy()

    Cu = (2.0 * deltat * Du) / (deltax**2)
    Cv = (2.0 * deltat * Dv) / (deltax**2)

    output_dir = 'frames_expansion'
    if plot_every:
        _save_frame(u, F, N, output_dir, 0)

    # Exit corner, symmetric to the start corner where the wave is seeded.
    # Watched only when stop_at_exit is enabled.
    exit_region = (slice(N - 55, N - 5), slice(N - 55, N - 5))
    stop_step = None  # set once the front is first detected at the exit

    for t in range(T):

        u_new, v_new = _step_dufort_frankel(
            u, v, u_old, v_old, u_new, v_new,
            F, Cu, Cv, deltat, epsilon, alpha, beta, N)

        # Neumann boundary conditions
        for arr in (u_new, v_new):
            arr[0, :] = arr[1, :]
            arr[-1, :] = arr[-2, :]
            arr[:, 0] = arr[:, 1]
            arr[:, -1] = arr[:, -2]

        # Leapfrog advance: old <- current, current <- average(current, new)
        np.copyto(u_old, u)
        np.copyto(v_old, v)
        np.add(u, u_new, out=u); u *= 0.5
        np.add(v, v_new, out=v); v *= 0.5

        # In phase 2, re-impose the wave generators at both corners
        if phase == 2:
            u[5:45, 5:45] = u_max
            u[N-45:N-5, N-45:N-5] = u_max
            v[5:45, 5:45] = v_max
            v[N-45:N-5, N-45:N-5] = v_max

        # Early stop once the wave front reaches the exit corner: avoids
        # running the remaining steps once the maze is already flooded.
        if stop_at_exit:
            if stop_step is None and (t + 1) % check_every == 0:
                if u[exit_region].max() > exit_threshold:
                    stop_step = t + exit_margin
            if stop_step is not None and t >= stop_step:
                print(f'Front reached the exit: stopping at step {t + 1}/{T}')
                break

        if plot_every and (t + 1) % plot_every == 0:
            _save_frame(u, F, N, output_dir, t + 1)
            print(f'Computed t={t + 1} and frame saved.')

    return u, v


def FTCS(u, v, u_max, v_max, u_min, v_min, deltat, deltax, N, T,
         epsilon, alpha, beta, Du, Dv, F, plot_every, phase):
    """Integrate the FitzHugh-Nagumo system with the FTCS scheme.

    Used in phase 2 (retraction): starting from the phase-1 final state, the
    wave retracts towards both corners and the surviving front traces the
    optimal path. FTCS (Forward-Time Central-Space) is a simple explicit
    scheme, adequate for the shorter retraction stage. The per-step update runs
    as a parallel Numba kernel.

    Parameters
    ----------
    See `dufort_frankel`; here `phase` tags the retraction (2).

    Returns
    -------
    u, v : np.ndarray
        Final activator and inhibitor fields.
    """
    u = np.ascontiguousarray(u, dtype=np.float64)
    v = np.ascontiguousarray(v, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.float64)

    u_n = u.copy()
    v_n = v.copy()
    u_np1 = u.copy()
    v_np1 = v.copy()

    inv_dx2 = 1.0 / (deltax**2)
    output_dir = 'frames_retraction'

    for t in range(T):

        u_np1, v_np1 = _step_ftcs(
            u_n, v_n, u_np1, v_np1,
            F, deltat, inv_dx2, epsilon, alpha, beta, Du, Dv, N)

        # Dirichlet generators at both corners
        u_np1[5:55, 5:55] = u_max
        u_np1[N-55:N-5, N-55:N-5] = u_max
        v_np1[5:55, 5:55] = v_max
        v_np1[N-55:N-5, N-55:N-5] = v_max

        # Swap buffers (no copy)
        u_n, u_np1 = u_np1, u_n
        v_n, v_np1 = v_np1, v_n

        if plot_every and (t + 1) % plot_every == 0:
            _save_frame(u_n, F, N, output_dir, t + 1)
            print(f'Computed t={t + 1} and frame saved.')

    return u_n, v_n
