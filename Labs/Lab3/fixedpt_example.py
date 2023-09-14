# import libraries
import numpy as np
def driver():
# test functions
    f1 = lambda x: x*(1+((7-x**5)/(x**2)))**3
# fixed point is alpha1 = 1.4987....
    f2 = lambda x: x-((7-x**5)/(x**2))
    f3 = lambda x: x - ((x**5)-7)/(5*x**4)
    f4 = lambda x: x - ((x**5)-7)/(12)
#fixed point is alpha2 = 3.09...
    Nmax = 100
    tol = 1e-10
# test f1 '''
    x0 = 7**(1/5)
    
    [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
    print("3a. function 1 only works with x0 = fxpt")
    print('\n')
#test f2 '''
    x0 = 7**(1/5)
    [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f2(xstar))
    print('Error message reads:',ier)
    print("3b. function 2 only works with x0 = fxpt")
    print('\n')
#f3
    x0 = 1
    
    [xstar,ier] = fixedpt(f3,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f3(xstar):',f3(xstar))
    print('Error message reads:',ier)
    print("3c. f3 works as expected")
    print('\n')
#f3
    x0 = 1
    
    [xstar,ier] = fixedpt(f4,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f4(xstar):',f4(xstar))
    print('Error message reads:',ier)
    print("3d. f4 itterated too many times but it does end up at the right place")
    print('\n')

    print("It doesnt converge for functions with too steep of a slope or too high of a derivative.",
          "For the first two functions, the slope is too steep and it diverges. The third behaves as we want. The fourth gets close but not close enough.")
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
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier]
driver()
