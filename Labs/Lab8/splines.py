import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
def driver():
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval = np.linspace(a,b,Neval)
    ''' number of intervals'''
    Nint = 10
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    yeval = eval_cub_spline(xeval,Neval,a,b,f,Nint)
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
        fex[j] = f(xeval[j])
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    plt.legend()
    #plt.show()
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show()


def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        '''temporarily store your info for creating a line in the interval of
        interest'''
        ind = np.where((xeval > xint[jint]) & (xeval < xint[jint +1]))
        n = len(ind[0])
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        for kk in range(n):
            '''use your line evaluator to evaluate the lines at each of the points
            in the interval'''
            '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with
            the points (a1,fa1) and (b1,fb1)'''
         
            yeval[ind[0][kk]] = eval_line(a1,fa1,b1,fb1,xeval[ind[0][kk]])

    return yeval

def eval_cub_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
    '''create vector to store the evaluation of the linear splines'''
    M= Ms(xint, f(xint), Nint)
    yeval = np.zeros(Neval)
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        '''temporarily store your info for creating a line in the interval of
        interest'''
        ind = np.where((xeval > xint[jint]) & (xeval < xint[jint +1]))
        n = len(ind[0])
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        for kk in range(n):
            '''use your line evaluator to evaluate the lines at each of the points
            in the interval'''
            '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with
            the points (a1,fa1) and (b1,fb1)'''
         
            yeval[ind[0][kk]] = eval_cubic(M[kk], M[kk+1], a1,b1, fa1, fb1,xeval[ind[0][kk]])
    return yeval

    
def eval_line(a1,fa1,b1,fb1, xpt):
    m = (fb1- fa1)/(b1-a1)
    line = lambda x : m*(x-a1)+fa1
    return line(xpt)
def eval_cubic(M1, M2, x1,x2, fx1, fx2, xpt):
    hi= x2 - x1
    C = fx1/hi - hi*M1/6
    D = fx2/hi - hi*M2/6

    Si = ((x2-xpt)**3)*M1/(6*hi) + ((xpt-x1)**3)*M2/(6*hi) + C*(x2-xpt) + D*(xpt-x1)
    return Si

def Ms(xint, yint, n):
    M = np.zeros((n-1,n-1))
    for i in range(n):
        for j in range(n):
            if (i==j):
                M[i][j] = 1/3
                if(j!=0):
                    M[i][j-1] = 1/12
                if(j!=n-1):
                    M[i][j+1] = 1/12
    y = np.zeros(n)
    h = xint[1] - xint[0]
    for i in range(1,n):
        if(i > 0 and i < n-1):
            y[i-1] = (yint[i+1]-2*yint[i]+yint[i-1])/(2*h**2)
    Minv = inv(M)

    return Minv@y

driver()