##3C

import numpy as np
import scipy.integrate as int
def driver():
    f = lambda x: 1/(1+x**2)
    a = -5
    b = 5
    n = 1292
    h = (b-a)/n
    x = np.linspace(a,b,n)
  

    quad_trap = trap(x, n, h, f)
    quad_simpson = simpson(x, n, h, f)
    quad_quad, err, info  = int.quad(f, a, b, full_output=1, epsrel=1e-4)
    print("n=1292")
    print("Trapazoid approximation:", quad_trap)
    print("Simpson's Approximation:", quad_simpson)
    print("quad function (tol=10^-4):", quad_quad, "n: ", info['neval'])
    quad_quad, err, info  = int.quad(f, a, b, full_output=1, epsrel=1e-6)
    print("quad function (tol=10^-6):", quad_quad, "n: ", info['neval'])


    n = 94
    h = (b-a)/n
    x = np.linspace(a,b,n)
    print("n=94")
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