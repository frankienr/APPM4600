# import libraries
import numpy as np
import matplotlib.pyplot as plt
def driver():
# test functions
    f1 = lambda x: (10/(x+4))**(1/2)


    Nmax = 20
    tol = 1e-10
# test f1 '''
    x0 = 1.5
    
    [xstar,ier,x, count] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
    x = x[1:count]
    diffs = convergence(x, xstar)
    #print(x)
    p = aitken(x, tol, Nmax)
    print(x)
    print(p)
    #print(diffs)
    

    


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
        
        if (abs(x1-x0)/x1 <tol):
            xstar = x1
            ier = 0
            print("itterations: ", count)
            return [xstar, ier, x, count]
        x0 = x1
        x[count+1] = x1
    xstar = x1
    ier = 1
    
    return [xstar, ier, x, count]

def convergence(vec, p):
    vec = vec - p
    p0 = vec[1]
    diffs = []
    for i in range(len(vec)-1):
        p1 = vec[i+1]
        diffs.append(abs(p1/p0))
        p0 = p1
    diffs = diffs[1:-1]
    #print (diffs)
    
    return diffs

def aitken(x, tol, Nmax):
    pHat = np.zeros((len(x), 1))
    for i in range(1, len(x)-3):
        pHat[i] = x[i]-((x[i+1]-x[i])**2)/(x[i+2]-2*x[i+1]+x[i])
        if(i>1):
            if(abs(pHat[i]-pHat[i-1]) < tol):
                return pHat

    return pHat
driver()
