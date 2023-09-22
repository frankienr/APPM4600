import numpy as np
import matplotlib.pyplot as plt
def driver():
    y = lambda x: x - 4*np.sin(2*x) - 3
    x = np.linspace(-2, 8, 100)

    yx = y(x)

    plt.plot(x, yx)
    plt.title("f(x) = x - 4sin(2x) - 3")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.show()

    f = lambda x: -1*np.sin(2*x) +5/4*x-3/4
    fx = f(x)
    plt.plot(x, fx)
    plt.plot(x,x)
    plt.title("fixed point equation")
    plt.grid(True, which='both')
    plt.show()

    Nmax = 100
    tol = 0.5*10**(-10)
# test f1 '''
    x0 = 1.8
    f1 =lambda x: -1*np.sin(2*x) +5/4*x-3/4
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
        x1 = f(x0)
        if (abs(x1-x0)/x1 <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]
driver()
