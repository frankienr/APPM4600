import numpy as np
import matplotlib.pyplot as plt
#from tabulate import tabulate

def driver():
    f = lambda x: x**6 - x - 1
    fp = lambda x: 6*x**5 - 1
    x0 = 2
    x1 = 1
    Nmax = 100
    tol = 10**-10
    
    (p,pstar,ier,it) = newton(f, fp, x0,tol, Nmax)
    print("Using Newton's method:")
    print("Root approx is:", pstar)
    print("Error message:", ier)
    er = error(p, pstar, it)
    print ("p_n   error")
    for i in range(it):
        print( p[i], er[i])
    plt.plot(np.log(abs(er[:-1])), np.log(abs(er[1:])))
    #plt.title("Newtons: |x_k+1 − α| vs |x_k − α|")
    plt.axis('scaled')
    plt.show()


    
    (p,pstar,ier,it) = secant(f, x0, x1, tol, Nmax)
    print("Using Secant method:")
    print("Root approx is:", pstar)
    print("Error message:", ier)
    er = error(p, pstar, it)
    print ("p_n   error")
    for i in range(it):
        print( p[i], er[i])

    plt.plot(np.log(abs(er[:-1])), np.log(abs(er[1:])))
    plt.title("Secant: |x_k+1 − α| vs |x_k − α|")
    plt.axis('scaled')
    plt.show()





def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
        f,fp - function and derivative
        p0 - initial guess for root
        tol - iteration stops when p_n,p_{n+1} are within tol
        Nmax - max number of iterations
    Returns:
        p - an array of the iterates
        pstar - the last iterate
        info - success message
            - 0 if we met tol
            - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it+1]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it+1]

def secant(f, p0, p1, tol, Nmax):
    p = np.zeros(Nmax+2)
    p[0] = p0
    p[1] = p1    
    if (f(p0) == 0):
        pstar = p0
        ier = 0
        return[p, pstar, ier, 0]
    
    if (f(p1) == 0):
        pstar = p1
        ier = 0
        return[p, pstar, ier, 0]
    
    fp0 = f(p0)
    fp1 = f(p1)

    it = 2
    while(it < Nmax):
        if((fp1 - fp0) == 0):
            ier = 1
            pstar = p0
            return[p, pstar, ier, it]
        
        p2 = p1 - (fp1*(p1-p0)/(fp1 - fp0))

        if(abs(p2-p1) < tol):
            ier = 0
            pstar = p2
            return[p, pstar, ier, it]
        
        p0 = p1
        fp0 = fp1
        p1 = p2
        fp1 = f(p2)
        p[it] = p2
        it = it + 1


def error(p, pstar, m):
    er = np.zeros(m)
    it = 0
    while(it < m):
        er[it] = p[it] - pstar
        it = it + 1

    return er




driver()