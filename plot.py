import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import networkx as nx 

def coupling(res, steps, N, C, K, alpha):
    element_avg = int(steps*0.5) # Number of elements in the averaging over time (r_avg).
    t = res.t # history of time
    theta_hist = res.y # history of theta
    op_hist = np.exp(1j * res.y).sum(axis=0) / N # history of order parameter
    r_hist = np.abs(op_hist) # history of r(t)
    psi_hist = np.angle(op_hist) # history of Psi(t)
    avg_r = np.sum(r_hist[:-element_avg])/element_avg # Average of last elements of r(t).
    plt.plot(t,r_hist)
    plt.xlabel('t',fontsize=18)
    plt.ylabel('r',fontsize=18)
    plt.title('Coupling constant for N, C, K = ' + str(N) + ', ' + str(C) + ', ' + str(K) + ' and α = ' + str(alpha))
    plt.show()

def avg_r(iterations, avg_r_hist):
    plt.plot(range(iterations), avg_r_hist)
    plt.xlabel('iteration',fontsize=18)
    plt.ylabel('avg_r',fontsize=18)
    plt.show()

def best_r(iterations, best_r_hist):
    plt.plot(range(iterations), best_r_hist)
    plt.xlabel('iteration',fontsize=18)
    plt.ylabel('best_r',fontsize=18)
    plt.show()

def graph(A, N, is_directed):
    G = nx.DiGraph() 
    for i in range(N):
        G.add_node(i+1)
        for j in range(N):
            if A[i][j] != 0: 
                G.add_edge(i+1, j+1, weight = A[i][j])

    plt.title('Graph displaying the oscillators and their respective connections.')
    if is_directed:
        nx.draw_shell(G, with_labels=True, font_weight='bold')
    else:
        nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
    #nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
    #nx.draw_planar(G, with_labels=True, font_weight='bold')
    #nx.draw_spring(G, with_labels=True, font_weight='bold')
    #nx.draw_shell(G, with_labels=True, font_weight='bold')
    plt.show()
    #plt.savefig("path.png")
    