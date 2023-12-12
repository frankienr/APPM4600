import numpy as np
import time
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

def driver():
    itterations = np.zeros(200)
    times = np.zeros(200)
    n = 400
    for c in range(10,200):
        print(c)
        temp = np.zeros(10)
        
        #for i in range(10):
        A = SPD(n,c)
        b = np.random.random(n)
        tol = 1e-5
        Nmax = 1000
        t = time.time()
        for j in range(50):
            (xstar, ier, its) = ConjGrad(A, b, n, Nmax, tol)
        t = time.time() - t
        
        #temp[i] = its
        #itteration = sum(temp)/10
        #itterations[c] = itteration
        times[c] = t
    #print(itterations)
    # plt.plot(range(5,200), itterations[5:])
    # plt.xlabel("Condition number")
    # plt.ylabel("Average itterations to converge")
    # plt.title("Effect of Condition Number on Iterations")
    # plt.show()
    plt.plot(range(10,200), times[10:])
    plt.xlabel("Condition number ")
    plt.ylabel("Time to compute")
    plt.title("Effect of Condition Number on Time")
    plt.show()


    
     



def SPD(n,c):
    Q,R = np.linalg.qr(np.random.random((n,n)))
    eig = np.linspace(1,c,n)
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