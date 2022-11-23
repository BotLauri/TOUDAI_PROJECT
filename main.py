import numpy as np
import random
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import plot
import optimize

# Start with small network. (N = 3, 4, 5)
# Change omega to uniformly spaced omega. 
# Minimum number of connections is N-1. 
# Connected network -> eigenvalue of the laplacian. Is connected one the best? Look at printscreen. 
# One 0 = connected. Look it up? 
# Link in slack. 

# Alpha = 0. C = 10/100. 

# General parameters.
N = 4 # Number of oscillators. 
C = int(0.25*N**2) # Number of connections. 
K = C*2 # Coupling constant. 
alpha = 0.0 # Phase shift. 
t_end = 200 # Calculation terminates at t = t_end.
steps = 1000 # Number of time steps in simulation. 
is_directed = True

# Initialization.
#seed = 2022 # Random seed. 
#random.seed(seed)
#rng = np.random.default_rng(seed=seed)
rng = np.random.default_rng()
# Initial values of omega follow normal distribution with mean = 0 and sd = 1.
#omega = rng.normal(loc=0, scale=1, size=N)
omega = np.linspace(0, 0.1*N, num=N)
# Initial values of phi follow uniform distribution. 
ic = rng.random(size=(N,)) * 2 * np.pi

# Optimization parameters. 
iterations = 100 # Optimization iterations. 
element_avg = int(steps*0.5) # Number of elements in the averaging over time (r_avg).
updates_per_iteration = int(C*0.2) # Number of connections removed/added in each iteration. 
if updates_per_iteration == 0:
    updates_per_iteration = 1
max_iterations_without_improvement = 5

# Initialize the network matrix. 
A = np.zeros((N, N))
c = 0
while c < C:
    x = rng.integers(0, N)
    y = rng.integers(0, N)
    if A[x][y] != 1:
        A[x][y] = 1
        c += 1

# TODO: MAKE A INIT SYMMETRIC AN OPTION.
def A_init(N):
    if is_directed:
        A = np.zeros((N, N))
        c = 0
        while c < C:
            x = rng.integers(0, N)
            y = rng.integers(0, N)
            if A[x][y] != 1:
                A[x][y] = 1
                c += 1
    else:
        A = np.zeros((N, N))
        c = 0
        while c < C:
            # TODO
            x = rng.integers(0, N)
            y = rng.integers(0, N)
    return A

# Calculate one step in the Kuramoto model. 
def kuramoto_network_model(t, phi, N, K, C, omega, alpha, A):
    phi_tile = np.tile(phi,(N, 1))
    sums = np.sum(A*(np.sin(phi_tile.T - phi_tile + alpha) - np.sin(alpha)), axis=0) # Is this phi_j - phi_i?
    phidot = omega + K/C*sums
    return phidot

# Simulation with elitism: If a number of iternations since last increase has passed, revert to old A. 
# TODO: ADD POSSIBILITY FOR SYMMETRIC A. 
def simulation(A, is_directed):
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
            for i in remove_connection:
                x, y = nonzero_indices[0][i], nonzero_indices[1][i]
                A[x][y] = 0
            for i in add_connection:
                x, y = zero_indices[0][i], zero_indices[1][i]
                A[x][y] = 1
        else:
            nonzero_indices = np.nonzero(A)

        it_nr = it_nr + 1
        avg_r_hist.append(avg_r)
        best_r_hist.append(best_r)

    return A, best_A, avg_r_hist, best_r_hist

#A, best_A, avg_r_hist, best_r_hist = simulation(A, is_directed)
A, best_A, avg_r_hist, best_r_hist = optimize.simulation(A, steps, C, t_end, kuramoto_network_model, N, K, omega, alpha, ic, is_directed)

# Do we have a connected network?
eigenvalues, eigenvectors = np.linalg.eig(best_A)
print(best_A)
print(eigenvalues)

# Solve the initial value problem for plotting. 
t_eval = np.linspace(0, t_end, steps)
res = solve_ivp(fun=kuramoto_network_model, args=(N, K, C, omega, alpha, best_A), y0=ic, 
                    t_span=(0, t_end), t_eval=t_eval, atol=1e-8, rtol=1e-8)
                        
#plot.coupling(res, element_avg, N, C, K, alpha)
#plot.avg_r(iterations, avg_r_hist)
#plot.best_r(iterations, best_r_hist)
