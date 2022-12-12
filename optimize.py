import numpy as np
import random
from scipy.integrate import solve_ivp

# Calculate one step in the Kuramoto model. 
def kuramoto_network_model(t, phi, N, K, C, omega, alpha, A):
    phi_tile = np.tile(phi,(N, 1))
    sums = np.sum(A*(np.sin(phi_tile.T - phi_tile + alpha) - np.sin(alpha)), axis=0)
    phidot = omega + K/C*sums
    return phidot

# Simulation with elitism: If a number of iternations since last increase has passed, revert to old A. 
def simulation(steps, t_end, C, N, K, omega, alpha, ic, iterations, is_directed, seed=None):
    # Optimization parameters. 
    element_avg = int(steps*0.5) # Number of elements in the averaging over time (r_avg).
    updates_per_iteration = int(C*0.2) # Number of connections removed/added in each iteration. 
    if updates_per_iteration == 0:
        updates_per_iteration = 1
    max_iterations_without_improvement = 10

    if seed is None:
        random.seed(seed)
        rng = np.random.default_rng(seed=seed)

    # TODO: Make the initialization smarter by only looking at zeroes. 
    # TODO: Make initialization into its own file?
    # Will make it faster in situations where almost all nodes have connections. 
    # Initialize the network matrix. Undirected graph is not allowed to have connections to the same node.
    if is_directed:
        A = np.zeros((N, N))
        c = 0
        while c < C:
            x = rng.integers(0, N)
            y = rng.integers(0, N)
            while x == y:
                y = rng.integers(0, N) # Make sure the connection is between different nodes. 
            if A[x][y] != 1:
                A[x][y] = 1
                c += 1
    else:
        if C % 2 == 1: # Can not have a half connection. 
            C += 1
        A = np.zeros((N, N))
        c = 0
        while c < C:
            x = rng.integers(0, N)
            y = rng.integers(0, N)
            while x == y:
                y = rng.integers(0, N) # Make sure the connection is between different nodes. 
            if A[x][y] != 1:
                A[x][y] = 1
                A[y][x] = 1
                c += 2

    best_r = 0
    it_nr = 0
    avg_r_hist = []
    best_r_hist = []
    for it in range(iterations):
        # If no improvement, revert to A. 
        if (it_nr > max_iterations_without_improvement):
            A = best_A.copy() # Deep copy. 
            it_nr = 0
            print(best_r)

        # Solve the initial value problem.
        t_eval = np.linspace(0, t_end, steps)
        res = solve_ivp(fun=kuramoto_network_model, args=(N, K, C, omega, alpha, A,), y0=ic, 
                        t_span=(0, t_end), t_eval=t_eval, atol=1e-8, rtol=1e-8)

        # Calculate avg_r. 
        op_hist = np.exp(1j * res.y).sum(axis=0) / N # history of order parameter.
        r_hist = np.abs(op_hist) # history of r(t).
        avg_r = np.sum(r_hist[:-element_avg])/element_avg # Average of last elements of r(t).

        # If best update accordingly. 
        if (avg_r > best_r):
            best_r = avg_r
            best_A = A.copy() # Deep copy. 
            it_nr = 0
            print(best_r)

        # Update the A matrix.
        if is_directed:
            nonzero_indices = np.nonzero(A)
            zero_indices = np.nonzero(A == 0) # Gives the zeroes instead. 
            # Choose a couple of connections to remove and add.
            remove_connection = random.sample(range(0, C), updates_per_iteration)
            add_connection = random.sample(range(0, N**2-C), updates_per_iteration)
            for i, j in zip(remove_connection, add_connection):
                x, y = nonzero_indices[0][i], nonzero_indices[1][i]
                a, b = zero_indices[0][i], zero_indices[1][i]
                if x != y and a != b:
                    A[x][y] = 0
                    A[a][b] = 1
        else:
            nonzero_indices = np.nonzero(np.triu(A, 1)) # Symmetry -> above the diagonal is enough. 
            zero_indices = np.nonzero(A == 0) # Gives the zeroes instead.
            temp = []
            for i, j in zip(zero_indices[0], zero_indices[1]): # Only care about zeroes above the diagonal.
                if i < j:
                    temp.append((i, j))
            zero_indices = temp
            # Choose a couple of connections to remove and add.
            remove_connection = random.sample(range(0, int(C/2)), updates_per_iteration)
            add_connection = random.sample(range(0, len(zero_indices)), updates_per_iteration)
            for i, j in zip(remove_connection, add_connection):
                x, y = nonzero_indices[0][i], nonzero_indices[1][i]
                a, b = zero_indices[j]
                A[x][y] = 0
                A[y][x] = 0 # Lower triangular matrix. 
                A[a][b] = 1
                A[b][a] = 1 # Lower triangular matrix. 

        it_nr = it_nr + 1
        avg_r_hist.append(avg_r)
        best_r_hist.append(best_r)

    return A, best_A, avg_r_hist, best_r_hist