import numpy as np
import time
import math
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    trials = 10
    differences = np.zeros(trials)
    for id in range(trials):
            
        n = 100
        A = SPD(n)
        Astar = A
        for i in range(2000):
            x = math.floor(n * np.random.random())
            y = math.floor(n * np.random.random())
            if(x == y):
                continue
            #Astar[x][y] = np.random.random()
            #print("Changed A at", x, y)
    
        b = np.random.random(n)
    
        #print("A:", A, "Astar:", Astar)
        tol = 1e-5
        Nmax = 1000
        tA = time.time()
        for i in range(50):
            (xstar, ier, its) = ConjGrad(A, b, n, Nmax, tol)
        tA = time.time() -tA
        print("Conjugate Gradient (symetric A):")
        print("Time:", tA, "Itterations:", its, "Message:", ier)

        tAstar = time.time()
        for i in range(50):
            (xstar, ier, its) = ConjGrad(Astar, b, n, Nmax, tol)
        tAstar = time.time() -tAstar
        print("Conjugate Gradient (non-symetric A):")
        print("Time:", tAstar, "Itterations:", its, "Message:", ier)
        differences[id] = ((tAstar - tA)/tA)*100

        #print(tA, tAstar)

    print(differences)
    print(sum(differences)/trials)
    
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


driver()