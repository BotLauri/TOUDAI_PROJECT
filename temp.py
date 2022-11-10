import numpy as np
import autograd.numpy as au
from autograd import grad, jacobian
from scipy.optimize import line_search

def func(x): # Objective function
    #return 4*x[0]**2 + 4*x[0]*x[1] + 5/2*x[1]**2 - 16*x[0] - 4*x[1]

Df = grad(func) # Gradient of the objective function

def Fletcher_Reeves(Xj, tol, alpha_1, alpha_2):
    x1 = [Xj[0]]
    x2 = [Xj[1]]
    D = Df(Xj)
    delta = -D # Initialize the descent direction
    
    while True:
        start_point = Xj # Start point for step length selection 
        beta = line_search(f=func, myfprime=Df, xk=start_point, pk=delta, c1=alpha_1, c2=alpha_2)[0] # Selecting the step length
        if beta!=None:
            X = Xj+ beta*delta #Newly updated experimental point
        
        if NORM(Df(X)) < tol:
            x1 += [X[0], ]
            x2 += [X[1], ]
            return X, func(X) # Return the results
        else:
            Xj = X
            d = D # Gradient at the preceding experimental point
            D = Df(Xj) # Gradient at the current experimental point
            chi = NORM(D)**2/NORM(d)**2 # Line (16) of the Fletcher-Reeves algorithm
            delta = -D + chi*delta # Newly updated descent direction
            x1 += [Xj[0], ]
            x2 += [Xj[1], ]

NORM = np.linalg.norm

Fletcher_Reeves(np.array([0., 0.]), 10**-5, 10**-4, 0.38)

