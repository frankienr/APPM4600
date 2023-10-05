import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm
def driver():
    x0 = np.array([1,1])
    Nmax = 100
    tol = 1e-10
   

    
    
   
    [xstar,ier,its,x] = LazyNewton(x0,tol,Nmax)
    print("1a")
    print("Root approx:", xstar)
    print("Itterations:", its)
    print("Error message:", ier)
    #print("Itternations:", x)

    [xstar,ier,its] = Newton(x0,tol,Nmax)
    print("1c")
    print("Root approx (Newton):", xstar)
    print("itterations:", its)
    print("Error message:", ier)
    
    
def evalF(x):
    F = np.zeros(2)
    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 -1
    return F

def evalJ(x):
    J = np.array([[(6*x[0]), (-2*x[1])],
                  [(3*x[1]**2 - 3*x[0]**2), (6*x[0]*x[1])]])
    return J

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


def LazyNewton(x0,tol,Nmax):
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
            ier =0
            return[xstar, ier,its, x]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its, x]
def Broyden(x0,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''
    '''Sherman-Morrison
    (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''
    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''
    '''In Broyden
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''
    ''' implemented as in equation (10.16) on page 650 of text'''
    '''initialize with 1 newton step'''
    A0 = evalJ(x0)
    v = evalF(x0)
    A = np.linalg.inv(A0)
    s = -A.dot(v)
    xk = x0+s
    for its in range(Nmax):
        '''(save v from previous step)'''
        w = v
        ''' create new v'''
        v = evalF(xk)
        '''y_k = F(xk)-F(xk-1)'''
        y = v-w;
        '''-A_{k-1}^{-1}y_k'''
        z = -A.dot(y)
        ''' p = s_k^tA_{k-1}^{-1}y_k'''
        p = -np.dot(s,z)
        u = np.dot(s,A)
        ''' A = A_k^{-1} via Morrison formula'''
        tmp = s+z
        tmp2 = np.outer(tmp,u)
        A = A+1./p*tmp2
        ''' -A_k^{-1}F(x_k)'''
        s = -A.dot(v)
        xk = xk+s
        if (norm(s)<tol):
            alpha = xk
            ier = 0
            return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]

driver()