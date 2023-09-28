import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x: (np.e**(3*x)) - 27*x**6 +(27*x**4)*(np.e**x) - (9*x**2)*(np.e**(2*x))
    fp = lambda x: (3*(np.e**(3*x))) - (162*x**5) + 27*((4*x**3)*(np.e**x) + (x**4)*(np.e**x)) - 9*(2*x*np.e**(2*x) + (2*x**2)*(np.e**(2*x)))
    g = lambda x: ((np.e**(3*x)) - 27*x**6 +(27*x**4)*(np.e**x) - (9*x**2)*(np.e**(2*x)))/(3*(np.e**(3*x)) - 162*x**5 + 27*((4*x**3)*(np.e**x) + (x**4)*(np.e**x)) - 9*(2*x*np.e**(2*x) + (2*x**2)*(np.e**(2*x))))
    gp = lambda x: ((x**2-4*x+2)*np.e**x + 6*x**2)/((np.e**x-6*x)**2)
    m = 2
    x = np.linspace(-1,4, 100)
    plt.ylim([-2, 2])
    plt.axhline(y=0, color='k')
    plt.title("f(x)")
    #plt.plot(x, fp(x))
    plt.plot(x, f(x))
    plt.show()
    p0 = 3
    tol = 10**-10
    Nmax = 100

    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print("Using normal Newton's")
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    er = error(p, pstar, it)
    print("Errors:", er)
    

    (p,pstar,info,it) = newton2(g,gp,p0,tol, Nmax)
    print("\nUsing g(x) = f(x)/f'(x)")
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    er = error(p, pstar, it)
    print("Errors:", er)
    

    (p,pstar,info,it) = newton3(f,fp,p0,tol, Nmax, m)
    print("\nUsing x_n+1 = x_n -mf(x)/f'(x)")
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    er = error(p, pstar, it)
    print("Errors:", er)
    
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

def newton2(g,gp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
        g,gp - function and derivative
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
        p1 = p0-g(p0)/gp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]


def newton3(f,fp,p0,tol,Nmax, m):
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
        p1 = p0-m*f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

def error(p, pstar, m):
    er = np.zeros(m)
    it = 0
    while(it < m):
        er[it] = p[it] - pstar
        it = it + 1

    return er

driver()