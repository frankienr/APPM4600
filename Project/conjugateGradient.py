import numpy as np
import math
import time
import random
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    n = 3
    A = SPD(n)
    # A = np.array([[1,0,0],
    #               [0,7, 0],
    #               [0,0, 9]])
    print(A)
    b = np.array([1,2,3])
    
    tol = 1e-5
    Nmax = 100

    (xstar, ier, its) = ConjGrad(A, b, n, Nmax, tol)
    print(xstar, ier, its)
    print(A@xstar)




def SPD(n):
    Q,R = np.linalg.qr(np.random.random((n,n)))
    eig = np.random.random(n)
    E = np.diag(eig)
    A = Q@E@np.transpose(Q)
    return A

    

def ConjGrad(A, b, n, Nmax, tol):
    x0 = np.zeros(n)
    r0 = b
    p0 = r0
    for k in range(1,Nmax):
        #step length
        alphak = (np.transpose(r0)@r0)/(np.transpose(p0)@A@p0)
        #approximated soln
        xk = x0 + alphak*p0
        #residual
        rk = r0 - alphak*A@p0
        #impovement in this step
        betak = (np.transpose(rk)@rk)/(np.transpose(r0)@r0)
        #new search direction
        pk = rk + betak*p0

        if((norm(xk-x0) < tol) or (norm(xk-x0)/norm(xk) < tol)):
            #success
            xstar = xk
            ier = 0
            return (xstar, ier, n)
        x0 = xk
        r0 = rk
        p0 = pk 

    ier = xk
    ier = 1
    return (xstar, ier, n)

driver()