import numpy as np
import random
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

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

# Optimization parameters. 
iterations = 100 # Optimization iterations. 
element_avg = int(steps*0.5) # Number of elements in the averaging over time (r_avg).
updates_per_iteration = int(C*0.2) # Number of connections removed/added in each iteration. 
if updates_per_iteration == 0:
    updates_per_iteration = 1
max_iterations_without_improvement = 5 # Self explanatory. 

# Initialization.
best_r = 0
#seed = 2022 # Random seed. 
#random.seed(seed)
#rng = np.random.default_rng(seed=seed)
rng = np.random.default_rng()
# Initial values of omega follow normal distribution with mean = 0 and sd = 1.
#omega = rng.normal(loc=0, scale=1, size=N)
omega = np.linspace(0, 0.5*N, num=N)
# Initial values of phi follow uniform distribution. 
ic = rng.random(size=(N,)) * 2 * np.pi

# Initialize the network matrix. 
A = np.zeros((N, N))
c = 0
while c < C:
    x = rng.integers(0, N)
    y = rng.integers(0, N)
    if A[x][y] != 1:
        A[x][y] = 1
        c += 1

def kuramoto_network_model(t, phi, N, K, C, omega, alpha, A):
    # Calculate one step in the Kuramoto model. 
    phi_tile = np.tile(phi,(N, 1))
    sums = np.sum(A*(np.sin(phi_tile.T - phi_tile + alpha) - np.sin(alpha)), axis=0) # Is this phi_j - phi_i?
    phidot = omega + K/C*sums
    return phidot

# Simulation with elitism: If a number of iternations since last increase has passed, revert to old A.  
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

    it_nr = it_nr + 1
    avg_r_hist.append(avg_r)
    best_r_hist.append(best_r)

# Do we have a connected network?
eigenvalues, eigenvectors = np.linalg.eig(A)
print(A)
print(eigenvalues)

# Solve the initial value problem for plotting. 
res = solve_ivp(fun=kuramoto_network_model, args=(N, K, C, omega, alpha, best_A), y0=ic, 
                    t_span=(0, t_end), t_eval=t_eval, atol=1e-8, rtol=1e-8)
t = res.t # history of time
theta_hist = res.y # history of theta
op_hist = np.exp(1j * res.y).sum(axis=0) / N # history of order parameter
r_hist = np.abs(op_hist) # history of r(t)
psi_hist = np.angle(op_hist) # history of Psi(t)
avg_r = np.sum(r_hist[:-element_avg])/element_avg # Average of last elements of r(t).
#print(avg_r)

plt.plot(t,r_hist)
plt.xlabel('t',fontsize=18)
plt.ylabel('r',fontsize=18)
plt.title('Coupling constant for N, C, K = ' + str(N) + ', ' + str(C) + ', ' + str(K) + ' and α = ' + str(alpha))
plt.show()

plt.plot(range(iterations), avg_r_hist)
plt.xlabel('iteration',fontsize=18)
plt.ylabel('avg_r',fontsize=18)
plt.show()

plt.plot(range(iterations), best_r_hist)
plt.xlabel('iteration',fontsize=18)
plt.ylabel('best_r',fontsize=18)
plt.show()
