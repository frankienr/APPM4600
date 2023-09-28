import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def driver():

    Ti = 20
    Ts = -15
    a = 0.138*10**(-6)
    tol= 10**-13
    Nmax = 100
    t = 60*60*24*60
    a0 = 0
    b0 = 2
    x0 = 0.01



    x = np.linspace(0, 2, 100)
    
    T = lambda x: (Ti-Ts)*special.erf(x/(2*np.sqrt(a*t))) + Ts
    Tp = lambda x: (Ti-Ts)*(2/np.sqrt(np.pi))*np.e**(-1*x**2)
    plt.plot(x, T(x))
    plt.axhline(y=0, color='k')
    plt.show()
    (pstar, ier) = bisection(T, a0, b0, tol)
    print("Root using bisection:", pstar)
    print("Error message", ier)

    (p,pstar,ier,it) = newton(T, Tp, x0, tol, Nmax)
    print("Root using Newton:", pstar)
    print("Error message", ier)

    #(p,pstar,ier,it) = newton(T, Tp, 2, tol, Nmax)
    #print("Root using Newton and x0 = 2:", pstar)
    #print("Error message", ier)



def bisection(f,a,b,tol):
    # Inputs:
        # f,a,b - function and endpoints of initial interval
        # tol - bisection stops when interval length < tol
    # Returns:
    # astar - approximation of root
    # ier - error message
    # - ier = 1 => Failed
    # - ier = 0 == success
    # first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]
    # verify end points are not a root
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]
    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]
    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
        fd = f(d)
        if (fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count = count +1
        # print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    return [astar, ier]

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
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]
driver()
