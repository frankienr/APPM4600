import numpy as np
def driver():
    f = lambda x: x**6 -x -1
    fp = lambda x: 6*x**5 - 1
    a= 0
    b = 3
    tol = 10**-10
    Nmax = 100
    (pstar, ier, it) = bisectionNewton(f, fp, a, b,tol, Nmax)
    print("root:", pstar)
    print("error message:", ier)

def bisectionNewton(f,fp,a,b,tol,Nmax):
    # Inputs:
        # f,a,b - function and endpoints of initial interval
        # tol - bisection stops when interval length < tol
    # Returns:
    # astar - approximation of root
    # ier - error message
    # - ier = 1 => Failed
    # - ier = 0 == success
    # first verify there is a root we can find in the interval
    g = lambda x: x - f(x)/fp(x)
    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]
    
    count = 0
    d = 0.5*(a+b)
    a0 = a
    b0 = b
    while ((g(d) > b0) or (g(d) < a0)):
    
        fd = f(d)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier, 0]

        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)

        count = count + 1
        
    
    
    for it in range(Nmax):
        p1 = d-f(d)/fp(d)
        
        if (abs(p1-d) < tol):
            pstar = p1
            info = 0
            return [pstar,info,it+1]
        d = p1
    pstar = p1
    info = 1
    return [pstar,info,it+1]


driver()