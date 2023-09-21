import numpy as np
import matplotlib.pyplot as plt
def driver():
    f = lambda x: (10/(x+4))**(1/2)
    p0 = 1.5
    Nmax = 20
    tol = 1e-10

    [pn, ier, Nmax, fpn, x] = steff(f, p0, tol, Nmax)

    print(x)





def steff(f, p0, tol, Nmax):
    x = np.zeros((Nmax,1))
    a = p0
    b = f(p0)
    c = f(b)
    for i in range(Nmax):
        pn = a - ((b-a)**2)/(c-2*b+a)
        x[i] = pn
        if(abs(pn-a) < tol):
            ier = 0
            return [pn, ier, i, f(pn), x]
        a = pn
        b = f(a)
        c = f(b)

        
    ier = 1
    return [a, ier, Nmax, f(a), x]

driver()