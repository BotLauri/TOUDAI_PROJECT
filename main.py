import numpy as np
import random
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import scipy
import plot
import optimize

# TODO: 
# Try all possibilites for N = 3,4. Find which setups are optimal. Different alpha=0.1, 0.2?
# Change omega to uniformly spaced omega. 
# Minimum number of connections is N-1. 
# Do we get series or branching network for N=4? (N!/2) Permutations but half are the same?
# Sort different networks by their optimal r values. 
# Later maybe look at C?
# Negative alpha -> nodes with most connections reverse order?
# Number of possible networks are: N!/2 + trees. (N!/2 are permutations minus mirror symmetry.)
# Analytical solution + Taylor?
# There are methods of adding/removing connections 
# which cause the best results, however, here we will just add randomly. 

# General parameters.
N = 7 # Number of oscillators. 
C = (N - 1)*2 # Number of connections. 
K = C*2 # Coupling constant. 
alpha = 0.5 # Phase shift. 
t_end = 200 # Calculation terminates at t = t_end.
steps = 1000 # Number of time steps in simulation. 
iterations = 50 # Optimization iterations. 
is_directed = False

# Initialization.
#seed = 2022
#random.seed(seed)
#rng = np.random.default_rng(seed=seed)
rng = np.random.default_rng()
# Initial values of omega follow normal distribution with mean = 0 and sd = 1.
#omega = rng.normal(loc=0, scale=1, size=N)
omega = np.linspace(0, 1, num=N)
# Initial values of phi follow uniform distribution. 
ic = rng.random(size=(N,)) * 2 * np.pi

# The code. There are parameters in the simulation function, these are not changed that often though. 
A, best_A, avg_r_hist, best_r_hist = optimize.simulation(steps, t_end, C, N, K, omega, alpha, ic, iterations, is_directed)

# Do we have a connected network? 
# If one eigenvalue is one (of the laplacian matrix) then we have a connected network.
laplacian = scipy.sparse.csgraph.laplacian(best_A)
eigenvalues, eigenvectors = np.linalg.eig(laplacian)
print(best_A)
print(eigenvalues)
#print(eigenvectors)

# Solve the initial value problem for plotting. 
t_eval = np.linspace(0, t_end, steps)
res = solve_ivp(fun=optimize.kuramoto_network_model, args=(N, K, C, omega, alpha, best_A), y0=ic, 
                    t_span=(0, t_end), t_eval=t_eval, atol=1e-8, rtol=1e-8)

#plot.coupling(res, steps, N, C, K, alpha)
#plot.avg_r(iterations, avg_r_hist)
#plot.best_r(iterations, best_r_hist)
plot.graph(best_A, N, is_directed)
