import matplotlib.pyplot as plt
import numpy as np
import time
import math
from numpy.linalg import inv
from numpy.linalg import norm
def driver():
    Nmax = 100
    x0= np.array([0,2,2])
    tol = 1e-6

    t = time.time()
    for j in range(50):
        [xstar, ier, its] = Newton(x0, tol, Nmax)
    elapsed = time.time() - t
    print("Newton found the root:", xstar)
    print("ier:", ier)
    print("iterations:", its)
    print("Time:", elapsed/50)

    t = time.time()
    for j in range(50):
        [xstar,gval,ier,its] = SteepestDescent(x0,tol,Nmax)
    elapsed = time.time() - t
    print("Steepest descent  found the root:",xstar)
    print("g evaluated at this point is ", gval)
    print("ier:", ier )
    print("iterations:", its)
    print("Time:", elapsed/50)

    tolS = 5 * 10**-2
    t = time.time()
    for j in range(50):
        [xstar,ier,its] = hybrid(x0, tolS, tol, Nmax)
    elapsed = time.time() - t
    print("Hybrid found the root:", xstar)
    print("ier:", ier)
    print("Itterations:",its)
    print("F(xstar) =",evalF(xstar))
    print("Time:", elapsed/50)
###########################################################
#functions:
def evalF(x):
    F = np.zeros(3)
    F[0] = x[0] +math.cos(x[0]*x[1]*x[2])-1.
    F[1] = (1.-x[0])**(0.25) + x[1] +0.05*x[2]**2 -0.15*x[2]-1
    F[2] = -x[0]**2-0.1*x[1]**2 +0.01*x[1]+x[2] -1
    return F
def evalJ(x):
    J = np.array([[1.+x[1]*x[2]*math.sin(x[0]*x[1]*x[2]),x[0]*x[2]*math.sin(x[0]*x[1]*x[2]
),x[1]*x[0]*math.sin(x[0]*x[1]*x[2])],
[-0.25*(1-x[0])**(-0.75),1,0.1*x[2]-0.15],
[-2*x[0],-0.2*x[1]+0.01,1]])
    return J
def evalg(x):
    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    gradg = np.transpose(J).dot(F)
    return gradg
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

###############################
### steepest descent code
def SteepestDescent(x,tol,Nmax):
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)
        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)
        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier,its]
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)
        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3
        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)
        if g0<=g3:
            alpha = alpha0
            gval = g0
        else:
            alpha = alpha3
            gval =g3
        x = x - alpha*z
        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier,its]
    print('max iterations exceeded')
    ier = 1
    return [x,g1,ier,Nmax]

def hybrid(x0, tolS, tol, Nmax):
    [xstar,gval,ier,it] = SteepestDescent(x0,tolS,Nmax)
    if (ier == 0):
        [xstar, ier, its] = Newton(xstar, tol, Nmax)
        return(xstar, ier, its+it)

    return(xstar, ier, -1)
driver()