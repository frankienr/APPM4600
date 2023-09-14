# import libraries
import numpy as np
def driver():
    # use routines
    print("solution for 1a")
    f = lambda x: (x**2)*(x-1)
    a = 0.5
    b = 2
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 10**(-5)
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print("root was found")

    print("solution for 1b")
    f = lambda x: (x**2)*(x-1)
    a = -1
    b = 0.5
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 10**(-5)
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print("no root was found")

    print("solution for 1c")
    f = lambda x: (x**2)*(x-1)
    a = -1
    b = 2
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 10**(-5)
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print("root was found")

    print("solution for 2a")
    f = lambda x: (x-1)*(x-3)*(x-5)
    a = 0
    b = 2.4
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 10**(-5)
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print("found a root")

    print("solution for 2b")
    f = lambda x: ((x-1)**2)*(x-3)
    a = 0
    b = 2
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 10**(-5)
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print("did not find a root")

    print("solution for 2c")
    f = lambda x: np.sin(x)
    a = 0.5
    b = 3*np.pi/4
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 10**(-5)
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))

    print('\n')
    print("these were all as I excepted. the ones that didn't provide a root made sense.")
# define routines
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
driver()

