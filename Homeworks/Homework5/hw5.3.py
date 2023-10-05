import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
def driver():
    x0 = np.array([1,1,1])
    Nmax = 100
    tol = 1e-10

    [xstar,ier,its] = Newton(x0,tol,Nmax)
    print("1c")
    print("Root approx (Newton):", xstar)
    print("itterations:", its)
    print("Error message:", ier)
    
def evalF(x):
    F = np.zeros(2)
    F[0] = x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16
    F[1] = 0
    
    return F

def evald(x):
    d = evalF(x)/()

def Newton(x0,tol,Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its]
