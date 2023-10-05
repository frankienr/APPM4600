import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm
def driver():
    x0 = np.array([1,0])
    Nmax = 100
    tol = 1e-10
   

    
    t = time.time()
    
    #[xstar,ier,its,x] = SlackerNewton(x0,tol,Nmax)
    elapsed = t - time.time()
    #print("Root approx:", xstar)
    #print("Itterations:", its)
    #print("Error message:", ier)
    #print("Time:", elapsed)
    #print("Itternations:", x)
    [xstar,ier,its] = approxNewton(x0,tol,Nmax)
    elapsed = t - time.time()
    print("Root approx:", xstar)
    print("Itterations:", its)
    print("Error message:", ier)
    #print("Time:", elapsed)
    #print("Itternations:", x)

    
    
    
def evalF(x):
    F = np.zeros(2)
    F[0] = 4*x[0]**2 +x[1]**2 -4
    F[1] = x[0] + x[1] - np.sin(x[0]- x[1])
    return F

def evalJ(x):
    J = np.array([[8*x[0], (2*x[1])],
                  [1-np.cos(x[0]- x[1]), 1+np.cos(x[0]- x[1])]])
    return J


def SlackerNewton(x0,tol,Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    J = evalJ(x0)
    Jinv = inv(J)
    x = []
    x = np.append(x, x0)
   
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        x = np.append(x, x1)
        if (norm(x1-x0) < tol):
            xstar = x1
            ier = 0
            return[xstar, ier,its, x]
       
        if(norm(x0 - x1) > 0.1):
            J = evalJ(x1)
            Jinv = inv(J)

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar,ier,its, x]

def approxJ(x, h):
    x0h = [x[0]+h, x[1]]
    x1h = [x[0], x[1]+h]
 
    J = np.array([[((4*x0h[0]**2 +x0h[1]**2 -4)-(4*x[0]**2 +x[1]**2 -4))/h, ((4*x1h[0]**2 +x1h[1]**2 -4)-(4*x[0]**2 +x[1]**2 -4))/h],
                  [((x0h[0] + x0h[1] - np.sin(x0h[0]- x0h[1]))- (x[0] + x[1] - np.sin(x[0]- x[1])))/h, ((x1h[0] + x1h[1] - np.sin(x1h[0]- x1h[1]))- (x[0] + x[1] - np.sin(x[0]- x[1])))/h]])
    return J

def approxNewton(x0,tol,Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    J = evalJ(x0)
    Jinv = inv(J) 
    h = 0.01
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its]
        x0 = x1
        J = approxJ(x0,h)
        Jinv = inv(J) 
    xstar = x1
    ier = 1
    return[xstar,ier,its]

driver()