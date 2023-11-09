####3A

import numpy as np
def driver():
    f = lambda x: 1/(1+x**2)
    a = -5
    b = 5
    n = 10
    h = (b-a)/n
    x = np.linspace(a,b,n)
  

    quad_trap = trap(x, n, h, f)
    quad_simpson = simpson(x, n, h, f)

    print("Trapazoid approximation:", quad_trap)
    print("Simpson's Approximation:", quad_simpson)



def trap(x, n, h, f):
    int = 1/2 * f(x[0])
    for i in range(1,n-1):
        int += f(x[i])

    int += 1/2 *f(x[n-1])
    int *= h

    return int

def simpson(x, n, h, f):
    int = 1/3 * f(x[0])
    n2 = round(n/2)
    for i in range(1, n2):
        int += 4/3 * f(x[2*i - 1])
        int += 2/3 * f(x[2*i])

    int += 1/3 * f(x[n-1])
    int *= h 
    return int


driver()