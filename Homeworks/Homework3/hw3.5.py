import numpy as np
import matplotlib.pyplot as plt
def driver():
    y = lambda x: x - 4*np.sin(2*x) - 3
    x = np.linspace(-2, 8, 100)

    yx = y(x)

    plt.plot(x, yx)
    plt.title("f(x) = x - 4sin(2x) - 3")
    plt.hlines(0,-2,8, 'black')
    plt.show()

    Nmax = 100
    tol = 1e-10
# test f1 '''
    x0 = 7**(1/5)
    
    [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)

# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    while (count <Nmax):
        #print("f(x) =", f(x0))
        count = count +1
        x1 = -1*np.sin(2*x0) +5*x0/4 - 3/4
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]
driver()
