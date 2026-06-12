import numpy as np
import matplotlib.pyplot as plt
import os


def dufort_frankel(u, v, u_max, v_max, u_min, v_min, deltat, deltax, N, T,
                   epsilon, alpha, beta, Du, Dv, F, plot_every, phase):
    """Integrate the FitzHugh-Nagumo system with the Dufort-Frankel scheme.

    Used in phase 1 (expansion): an autowave is launched from the start corner
    and floods the maze along the corridors. Dufort-Frankel is an explicit,
    unconditionally stable leapfrog scheme, which lets the front propagate over
    the long times the expansion needs.

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
        Save a frame every `plot_every` steps.
    phase : int
        Phase tag (1 = expansion); kept for symmetry with FTCS.

    Returns
    -------
    u, v : np.ndarray
        Final activator and inhibitor fields.
    """
    u_new = np.copy(u)
    v_new = np.copy(v)
    u_old = np.copy(u)
    v_old = np.copy(v)

    # Dufort-Frankel constants
    Cu = (2.0 * deltat * Du) / (deltax**2)
    Cv = (2.0 * deltat * Dv) / (deltax**2)

    # Plot setup
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(u, cmap='magma', vmin=-1.5, vmax=1.5, origin='lower')

    # Draw the maze walls as an overlay
    wall_overlay = np.zeros((N + 1, N + 1, 4))
    is_wall = (F < 0)
    wall_overlay[is_wall] = [0.8, 0.8, 0.8, 0.8]
    ax.imshow(wall_overlay, origin='lower')

    output_dir = 'frames_expansion'
    os.makedirs(output_dir, exist_ok=True)

    for t in range(T):

        u_old_c = u_old[1:N, 1:N]
        v_old_c = v_old[1:N, 1:N]
        F_c = F[1:N, 1:N]

        # Sum of neighbours
        sum_u = u[2:N+1, 1:N] + u[0:N-1, 1:N] + u[1:N, 2:N+1] + u[1:N, 0:N-1]
        sum_v = v[2:N+1, 1:N] + v[0:N-1, 1:N] + v[1:N, 2:N+1] + v[1:N, 0:N-1]

        # Kinetic terms
        react_u = epsilon * (u_old_c - u_old_c**3 - v_old_c + F_c)
        react_v = u_old_c - alpha * v_old_c + beta

        # Next time level
        u_new[1:N, 1:N] = (2.0 * deltat * react_u + u_old_c * (1.0 - 2.0 * Cu) + Cu * sum_u) / (1.0 + 2.0 * Cu)
        v_new[1:N, 1:N] = (2.0 * deltat * react_v + v_old_c * (1.0 - 2.0 * Cv) + Cv * sum_v) / (1.0 + 2.0 * Cv)

        # Neumann boundary conditions
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]

        v_new[0, :] = v_new[1, :]
        v_new[-1, :] = v_new[-2, :]
        v_new[:, 0] = v_new[:, 1]
        v_new[:, -1] = v_new[:, -2]

        # In phase 2, force the wave generators at both corners
        if phase == 2:
            u_new[5:45, 5:45] = u_max
            u_new[N-45:N-5, N-45:N-5] = u_max
            v_new[5:45, 5:45] = v_max
            v_new[N-45:N-5, N-45:N-5] = v_max

        # Advance time
        u_old = np.copy(u)
        v_old = np.copy(v)

        u = (np.copy(u_new) + u) / 2.0
        v = (np.copy(v_new) + v) / 2.0

        # Re-impose the generators after the average
        if phase == 2:
            u[5:45, 5:45] = u_max
            u[N-45:N-5, N-45:N-5] = u_max
            v[5:45, 5:45] = v_max
            v[N-45:N-5, N-45:N-5] = v_max

        # Save a frame
        if t % plot_every == 0:
            image.set_data(u)
            ax.set_title(f'Time step: {t}')
            frame_path = os.path.join(output_dir, f'frame_{t}.png')
            plt.savefig(frame_path)
            print(f'Computed t={t} and frame saved.')

    return u, v


def FTCS(u, v, u_max, v_max, u_min, v_min, deltat, deltax, N, T,
         epsilon, alpha, beta, Du, Dv, F, plot_every, phase):
    """Integrate the FitzHugh-Nagumo system with the FTCS scheme.

    Used in phase 2 (retraction): starting from the phase-1 final state, the
    wave retracts towards both corners and the surviving front traces the
    optimal path. FTCS (Forward-Time Central-Space) is a simple explicit
    scheme, adequate for the shorter retraction stage.

    Parameters
    ----------
    See `dufort_frankel`; here `phase` tags the retraction (2).

    Returns
    -------
    u, v : np.ndarray
        Final activator and inhibitor fields.
    """
    u_new = np.copy(u)
    v_new = np.copy(v)

    # Plot setup
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(u, cmap='magma', vmin=-1.5, vmax=1.5, origin='lower')

    # Draw the maze walls as an overlay
    wall_overlay = np.zeros((N + 1, N + 1, 4))
    is_wall = (F < 0)
    wall_overlay[is_wall] = [0.8, 0.8, 0.8, 0.8]
    ax.imshow(wall_overlay, origin='lower')

    output_dir = 'frames_retraction'
    os.makedirs(output_dir, exist_ok=True)

    for t in range(T):

        # Interior (everything but the borders), equivalent to range(1, N)
        u_c = u[1:N, 1:N]
        v_c = v[1:N, 1:N]
        F_c = F[1:N, 1:N]

        # Vectorised Laplacians
        lap_u = (u[2:N+1, 1:N] + u[0:N-1, 1:N] + u[1:N, 2:N+1] + u[1:N, 0:N-1] - 4*u_c) / deltax**2
        lap_v = (v[2:N+1, 1:N] + v[0:N-1, 1:N] + v[1:N, 2:N+1] + v[1:N, 0:N-1] - 4*v_c) / deltax**2

        # Update the interior
        u_new[1:N, 1:N] = u_c + deltat * (epsilon * (u_c - u_c**3 - v_c + F_c) + Du * lap_u)
        v_new[1:N, 1:N] = v_c + deltat * (u_c - alpha * v_c + beta + Dv * lap_v)

        # Dirichlet generators at both corners
        u_new[5:55, 5:55] = u_max
        u_new[N-55:N-5, N-55:N-5] = u_max
        v_new[5:55, 5:55] = v_max
        v_new[N-55:N-5, N-55:N-5] = v_max

        # Store u and v
        u = np.copy(u_new)
        v = np.copy(v_new)

        if t % plot_every == 0:
            image.set_data(u)
            ax.set_title(f'Time step: {t}')
            frame_path = os.path.join(output_dir, f'frame_{t}.png')
            plt.savefig(frame_path)
            print(f'Computed t={t} and frame saved.')

    return u, v
