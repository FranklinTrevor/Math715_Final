import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_solution(u, x, u_exact, x_exact, dt=1.0, fps=60, speed=1.0):
    nt, nx = u.shape
    t_end = (nt - 1) * dt

    # How many frames we *can* show during playback
    n_show = max(2, int(np.ceil((t_end / speed) * fps)))

    # Pick evenly-spaced frame indices to show
    frame_idx = np.linspace(0, nt - 1, n_show).astype(int)

    # Interval in ms to make total duration match (t_end / speed)
    interval = 1000 * (t_end / speed) / n_show

    fig, ax = plt.subplots()
    line,  = ax.plot(x, u[0], lw=2, color='black', label='Numerical')
    line2, = ax.plot(x_exact, u_exact[0], lw=2, color='red', linestyle='--', label='Exact')

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u(x,t)", fontsize=12)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.min(u), np.max(u))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    def update(k):
        n = frame_idx[k]
        line.set_ydata(u[n])
        line2.set_ydata(u_exact[n])
        ax.set_title(f"t = {n*dt:.4f} s")
        return line, line2

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_idx),
        interval=interval,
        blit=False,
        repeat=False
    )

    plt.show()
    return ani

def sparse_view(x):
    plt.figure(figsize=(5, 5))
    plt.spy(x, markersize=8, marker='o', color = 'Black')  
    plt.show()

def view_basis_sparse(T):
    len_E = len(T["Elements"])
    len_N = 3

    #Basis 1
    basis_matrix_1 = np.zeros((len_N,len_E))

    basis_matrix_1[1,0] = 1 #Elm 1
    basis_matrix_1[2,1] = 1 #Elm 2
    basis_matrix_1[2,2] = 1 #Elm 3
    basis_matrix_1[1,7] = 1 #Elm 8
    basis_matrix_1[0,8] = 1 #Elm 9
    basis_matrix_1[0,9] = 1 #Elm 10

    #Basis 3
    basis_matrix_2 = np.zeros((len_N,len_E))

    basis_matrix_2[1,4] = 1 #Elm 5
    basis_matrix_2[2,5] = 1 #Elm 6
    basis_matrix_2[1,11] = 1 #Elm 12

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4)) # figsize adjusts the overall size

    ax1.spy(basis_matrix_1, markersize=8, marker='o', color = 'Black')  
    ax1.set_title('Basis Functions for 1st Free node (Node 6)')
    ax1.set_xlabel('Element')
    ax1.set_ylabel('Node on Element')

    ax2.spy(basis_matrix_2, markersize=8, marker='o', color = 'Black')
    ax2.set_title('Basis Functions for 3rd Free node (Node 8)')
    ax2.set_xlabel('Element')
    ax2.set_ylabel('Node on Element')

    plt.tight_layout()

    plt.show()




