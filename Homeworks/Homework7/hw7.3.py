import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1/(1+(10*x)**2)
    N = 10

    h = 2/(N-1)
    ''' interval'''
    a = -1
    b = 1
    
    xint = np.zeros(N)
    for i in range(1,N+1):
        xint[i-1] = np.cos(2*i - 1)*np.pi/(2*N)

  
    ''' create interpolation data'''
    yint = f(xint)
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1001
    xeval = np.linspace(a,b,Neval+1)
    
    yeval_m = np.zeros(Neval+1)
    

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_m[kk] = barycentric(xint, yint, N, xeval[kk])

    ''' create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,'o')
    plt.plot(xeval,yeval_m,'r')
    plt.title("N=10")

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
    

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_m[kk] = barycentric(xint, yint, N, xeval[kk])

    ''' create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,'o')
    plt.plot(xeval,yeval_m,'r')
    plt.title("N=20")

    N = 100

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
    

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_m[kk] = barycentric(xint, yint, N, xeval[kk])

    ''' create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,'o')
    plt.plot(xeval,yeval_m,'r')
    plt.title("N=100")

    plt.figure()
    err_m = abs(yeval_m-fex)
    plt.semilogy(xeval,err_m,'g--',label='Error, N=100')
    plt.legend()

    plt.show()
def wj(xint, n, j):
    temp = 1 
    for i in range(n):
        if(i != j):
            temp = temp * (xint[j] - xint[i])

    return (1/temp)

def phin(xpt, xint, n):
    phi = 1
    for i in range(n):
        phi = phi * (xpt - xint[i])

    return phi

def barycentric(xint, yint, n, xpt):
    s = 0
    for j in range(n):

        s += (wj(xint, n, j)*yint[j])/(xpt - xint[j])

    return (phin(xpt, xint, n) * s)


driver()