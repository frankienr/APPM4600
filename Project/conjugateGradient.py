import numpy as np
import time
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve

def driver():
    n = 100
    A = SPD(n)
    b = np.random.random(n)
    x = np.zeros(n)
    f = lambda x: norm(x, A@x) - 2*norm(x, b)
    #print("A:", A, "b:", b)
    tol = 1e-5
    Nmax = 1000
    t = time.time()
    for i in range(50):
        (xstar, ier, its) = ConjGrad(A, b, n, Nmax, tol)
    t = time.time() -t
    print("Conjugate Gradient:")
    print("Time:", t, "Itterations:", its, "Message:", ier)
    #print("Conjugate Gradient soln:", xstar, "Itterations:", its, "Message:", ier)
    #print(A@xstar - b)
    t= time.time()
    for i in range(50):
        l, u = lu_factor(A)
        xstar = lu_solve((l,u),b)
    t = time.time() - t
    print("Guassian Elimination:")
    print("Time:", t)
    #print("Guassian Elimination soln:", xstar)
    t= time.time()
    for i in range(50):
        (xstar, its, ier) = steepest_descent(A, b, x, tol, Nmax)
    t = time.time() - t
    print("Steepest Descent:")
    print("Time:", t, "Itterations:", its, "Message:", ier)
    #print("Steepest Descent soln:", xstar, "Itterations:", its, "Message:", ier)




def SPD(n):
    Q,R = np.linalg.qr(np.random.random((n,n)))
    kon = 10
    eig = np.linspace(1,kon,n)
    E = np.diag(eig)
    A = Q@E@np.transpose(Q)
    return A

    

def ConjGrad(A, b, n, Nmax, tol):
    x0 = np.zeros(n)
    r0 = b
    p0 = r0
    for k in range(0,Nmax):
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

        if((norm(xk-x0)/norm(xk) < tol)):
            #success
            xstar = xk
            ier = 0
            return (xstar, ier, k)
        x0 = xk
        r0 = rk
        p0 = pk 

    xstar = xk
    ier = 1
    return (xstar, ier, n)


def steepest_descent(A, b, x, tol, Nmax):
    """
    Solve Ax = b
    Parameter x: initial values
    """
    r = b - A @ x
   
    for k in range(0, Nmax):
        p = r
        q = A @ p
        alpha = (p @ r) / (p @ q)
        x = x + alpha * p
        r0 = r
        r = r - alpha * q
        if(norm(r-r0) < tol):
            ier = 0
            return (x, k, ier)

       
    ier = 1
    k = Nmax
    return (x, k, ier)

driver()