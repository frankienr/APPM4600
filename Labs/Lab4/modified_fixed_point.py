# import libraries
import numpy as np
def driver():
# test functions
    f1 = lambda x: x - ((x**5)-7)/(5*x**4)


    Nmax = 10
    tol = 1e-10
# test f1 '''
    x0 = 1
    
    [xstar,ier,x] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
    for i in range(1,Nmax):
        print(i, x[i])


# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    x = np.zeros((Nmax,1))
    x[1] = x0
    while (count <Nmax):
    
        count = count +1
        x1 = f(x0)
        
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar, ier, x]
        x0 = x1
        x[count+1] = x1
    xstar = x1
    ier = 1
    return [xstar, ier, x]
driver()
