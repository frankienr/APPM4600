import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.linalg import lu_factor, lu_solve
def driver():
    # dts = np.linspace(0.05,0.5, 10)
    # timesCG = np.zeros(len(dts))
    # timesLU = np.zeros(len(dts))

    # for j, dt in enumerate(dts):
    dt = 0.1
    # print(dt)
    f = lambda x: np.exp(x)
    start=0
    end=10
    xvals = np.arange(start,end, dt)
    m = 1
    k = 4
    
    tol = 1e-5
    Nmax = 1000
    A = np.zeros((len(xvals), len(xvals)))
    b = np.zeros(len(xvals))

    for idx, x in enumerate(xvals):
    # print("idx", idx)
        if idx == 0:
            #boundary condition
            A[idx,idx] = 1
            b[idx] = 0
        elif idx == len(xvals) - 1:
            #boundary conditions
            A[idx,idx] = 1
            b[idx] = 0 
        else:
            #using the centered differences 
            #m((y_i+1-2y_i+y_i-1)/dt^2)+ky_i=sinx
            A[idx,idx - 1] = m/(dt**2)
            A[idx,idx] = -2*m/(dt**2) + k
            A[idx,idx + 1] = m/(dt**2)
            b[idx] = f(x)


    # print(A,b)
    t = time.time()
    for i in range(50):
        (ystar, ier, its) = ConjGrad(A, b, len(xvals), Nmax, tol)
    t = time.time() - t
    print("CG done")
    print("Conjugate Gradient:")
    print("Time:", t, "Itterations:", its, "Message:", ier)
    plt.plot(xvals, ystar, '-', label='Approximation with CG, dt = 0.1')
    print("CG:", ystar)
    # plt.plot(xvals, ystar)
    # timesCG[j] = t
    t= time.time()
    for i in range(50):
        l, u = lu_factor(A)
        ystar2 = lu_solve((l,u),b)
    t = time.time() - t
    print("LU done")
    print("Guassian Elimination:")
    print("Time:", t)
    plt.plot(xvals, ystar2, 'r--', label='Approximation with LU, dt = 0.1')
    print("LU:", ystar2)
    print(norm(ystar-ystar2))
    # timesLU[j] = t

    plt.legend()
    plt.title("Approximating y''+4y=e^x")
    plt.grid()
    plt.show()
    # plt.plot(dts, timesCG, label='Conjugate gradient')
    # plt.plot(dts, timesLU, label='Guassian elimination')
    # plt.legend()
    # plt.title("Approximation of y''+4y=e^x, decreasing dt")
    # plt.show()


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
    return (xstar, ier, Nmax)

driver()