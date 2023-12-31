import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1/(1+(10*x)**2)
    N = 20
    h = 2/(N-1)
    ''' interval'''
    a = -1
    b = 1
    
    xint = np.zeros(N)
    for i in range(1,N+1):
        xint[i-1] = -1 + (i-1)*h

  
    ''' create interpolation data'''
    yint = f(xint)
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1001
    xeval = np.linspace(a,b,Neval+1)
    
    yeval_m = np.zeros(Neval+1)
    coefs = vander(xint, N, yint)

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_m[kk] = evalPoly(coefs, xeval[kk], N)

    ''' create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,'g')
    plt.plot(xeval,yeval_m,'r')
    plt.plot(xint, yint, 'o')
    plt.title('N = %i' %N)
    
    plt.figure()
    err_m = abs(yeval_m-fex)
    plt.semilogy(xeval,err_m,'g',label='Error')
    plt.title('N = %i' %N)
    plt.legend()
   
    plt.show()
   


def vander(xint, n, yint):
    van = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            van[i][j] = xint[i] ** (n-(j+1))
   
    vaninv = la.inv(van)
    coefs = (vaninv @ yint)

    return coefs

def evalPoly(coefs, xpt, n):
    x = 0
    for i in range(n):
        x += coefs[i] * (xpt ** (n-(i+1)))

    return x

driver()