# All 4 node connected networks.
import numpy as np
from scipy.integrate import solve_ivp
from optimize import kuramoto_network_model
import plot

# General parameters.
N = 4 # Number of oscillators. 
C = 3 # Number of connections. 
K = C*2 # Coupling constant. 
alpha = 0.0 # Phase shift. 
t_end = 200 # Calculation terminates at t = t_end.
steps = 1000 # Number of time steps in simulation. 
element_avg = int(steps*0.5) # Number of elements in the averaging over time (r_avg).
is_directed = False

# Initialization.
seed = 2022
rng = np.random.default_rng(seed=seed)
omega = np.linspace(0, 1, num=N)
# Initial values of phi follow uniform distribution. 
ic = rng.random(size=(N,)) * 2 * np.pi

# The possible networks.
As = []
As.append([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]) # Tree, root 1. Picture 2. 
As.append([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]]) # Tree, root 2. Picture 11.
As.append([[0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]) # Tree, root 3. Picture 6.
As.append([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]]) # Tree, root 4. Picture 10.
As.append([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]) # Right Snake, 1-2-3-4/4-3-2-1. Picture 4. 
As.append([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]) # Up Snake, 4-1-2-3/3-2-1-4. Picture 1.
As.append([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]]) # Left Snake, 3-4-1-2/2-1-4-3. Picture 5.
As.append([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]) # Down Snake, 2-3-4-1/1-4-3-2. Picture 8.
As.append([[0, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]]) # Right X, 1-3-2-4/4-2-3-1. Picture 13. 
As.append([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]) # Down X, 1-3-4-2/2-4-3-1. Picture 14. 
As.append([[0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0]]) # Left X, 2-4-1-3/3-1-4-2. Picture 15. 
As.append([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]) # Up X, 3-1-2-4/4-2-1-3. Picture 16. 
As.append([[0, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]]) # N, 4-1-3-2/2-3-1-4. Picture 3.
As.append([[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 0]]) # Z, 1-2-4-3/3-4-2-1. Picture 9.
As.append([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]]) # Reverse N, 1-4-2-3/3-2-4-1. Picture 12.
As.append([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]]) # Reverse Z, 2-1-3-4/4-3-1-2. Picture 7.
lengths = [3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

avg_r = []
for A in As:
    # Solve the initial value problem.
    t_eval = np.linspace(0, t_end, steps)
    res = solve_ivp(fun=kuramoto_network_model, args=(N, K, C, omega, alpha, A), y0=ic, 
                    t_span=(0, t_end), t_eval=t_eval, atol=1e-8, rtol=1e-8)

    # Calculate avg_r. 
    op_hist = np.exp(1j * res.y).sum(axis=0) / N # history of order parameter.
    r_hist = np.abs(op_hist) # history of r(t).
    #avg_r.append(round(np.sum(r_hist[:-element_avg])/element_avg, 2)) # Average of last elements of r(t).
    avg_r.append(np.sum(r_hist[:-element_avg])/element_avg) # Average of last elements of r(t).

    #plot.graph(A, N, is_directed)

print(avg_r)
print(lengths)
