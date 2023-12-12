import numpy as np
import time
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

def driver():
    c=10
    ns = np.arange(100,450,10)
    timeCG = np.zeros(len(ns))
    timeSD = np.zeros(len(ns))
    timeLU = np.zeros(len(ns))
    for idx,n in enumerate(ns):
        print(n)
        for j in range(5):
            A = SPD(n, c)
            b = np.random.random(n)
            x = np.zeros(n)
            tol = 1e-5
            Nmax = 1000
            t = time.time()
            for i in range(50):
                (xstar, ier, its) = ConjGrad(A, b, n, Nmax, tol)
            t = time.time() -t
            timeCG[idx] += t
            # print("Conjugate Gradient:")
            # print("Time:", t, "Itterations:", its, "Message:", ier)
            #print("Conjugate Gradient soln:", xstar, "Itterations:", its, "Message:", ier)
            #print(A@xstar - b)
            t= time.time()
            for i in range(50):
                l, u = lu_factor(A)
                xstar = lu_solve((l,u),b)
            t = time.time() - t
            timeLU[idx] += t
            # print("Guassian Elimination:")
            # print("Time:", t)
            #print("Guassian Elimination soln:", xstar)
            t= time.time()
            for i in range(50):
                (xstar, its, ier) = steepest_descent(A, b, x, tol, Nmax)
            t = time.time() - t
            timeSD[idx] += t
            # print("Steepest Descent:")
            # print("Time:", t, "Itterations:", its, "Message:", ier)
            #print("Steepest Descent soln:", xstar, "Itterations:", its, "Message:", ier)
    timeCG = timeCG/5
    timeLU = timeLU/5
    timeSD = timeSD/5
    plt.plot(ns,timeCG, label = "Conjugate Gradient")
    plt.plot(ns,timeLU, label = "Gaussian Elimination")
    plt.plot(ns,timeSD, label = "Steepest Descent")
    plt.legend()
    plt.xlabel("Size of A")
    plt.ylabel("Time to compute")
    plt.title("Effect of size on time")
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