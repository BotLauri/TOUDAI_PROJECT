import numpy as np
import cmath
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

# Parameters.
N = 5 # Number of oscillators.
C = 10 # Number of connections. 
K = 1 # Coupling constant.
seed = 2022 # Random seed.
alpha = 0.3 # Phase shift. 

# Initialization.
rng = np.random.default_rng(seed=seed) # Generator. 
# Initial values of omega follow normal distribution with mean = 0 and sd = 1.
omega = rng.normal(loc=0, scale=1, size=N)
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

def kuramoto_network_model(t, phi, N, K, C, omega, alpha):
    # Calculate one step in the Kuramoto model. 
    phidot = np.zeros(N)
    sum = np.zeros(N)
    for i in range(N):
        for j in range(N):
            sum[i] += A[i][j] * cmath.sin(phi[j] - phi[i] + alpha)
    phidot[i] = omega[i] + K/C * sum[i]
    return phidot

# Simulation. 
t_end = 1000 # Calculation terminates at t = t_end.
steps = 100
t_eval = np.linspace(0, t_end, steps)
# Solve the initial value problem.
res = solve_ivp(fun=kuramoto_network_model, args=(N, K, C, omega, alpha,), y0=ic, 
                t_span=(0, t_end), t_eval=t_eval, atol=1e-6, rtol=1e-3)

print(omega)
print(ic)
print(A)
