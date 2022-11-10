import numpy as np
import cmath
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

# Parameters.
N = 100 # Number of oscillators. 
C = 2500 # Number of connections. 
K = 5000 # Coupling constant. 
seed = 2022 # Random seed. 
alpha = 0.1 # Phase shift. 

# Initialization.
rng = np.random.default_rng(seed=seed) # Generator. 
# Initial values of omega follow normal distribution with mean = 0 and sd = 1.
omega = rng.normal(loc=0, scale=1, size=N)
# Initial values of phi follow uniform distribution.
ic = rng.random(size=(N,)) * 2 * np.pi

# Initialize the network matrix. 
# Maybe change to list of tuples. Should be a lot quicker. Scrap?
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
            sum[i] += A[i][j] * np.sin(phi[j] - phi[i] + alpha)
        phidot[i] = omega[i] + K/C * sum[i]
    return phidot

def kuramoto_network_model_2(t, phi, N, K, C, omega, alpha):
    # Calculate one step in the Kuramoto model. 
    phi_tile = np.tile(phi,(N, 1))
    sums = np.sum(A*np.sin(phi_tile.T - phi_tile + alpha), axis=0) # Is this phi_j - phi_i?
    phidot = omega + K/C*sums
    return phidot

# Simulation. 
t_end = 10 # Calculation terminates at t = t_end.
steps = 1000
t_eval = np.linspace(0, t_end, steps)
# Solve the initial value problem.
res = solve_ivp(fun=kuramoto_network_model_2, args=(N, K, C, omega, alpha,), y0=ic, 
                t_span=(0, t_end), t_eval=t_eval, atol=1e-6, rtol=1e-3)

# Prints. 
#print(A)
#print(res)

# Stolen code^TM.
t = res.t # history of time
theta_hist = res.y # history of theta
op_hist = np.exp(1j * res.y).sum(axis=0) / N # history of order parameter
r_hist = np.abs(op_hist) # history of r(t)
psi_hist = np.angle(op_hist) # history of Psi(t)

plt.plot(t,r_hist)
plt.xlabel('t',fontsize=18)
plt.ylabel('r',fontsize=18)
plt.title('Coupling constant for α = ' + str(alpha))
plt.show()
