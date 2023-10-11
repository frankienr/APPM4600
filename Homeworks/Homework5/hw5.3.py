import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt
def driver():
    x0 = np.array([1,1,1])
    tol = 10**(-10)
    Nmax = 100
    x = []
    x = np.append(x, x0)
    i = 1
    while(i < Nmax):
        x1 = np.zeros(3)
        x1[0] = x0[0] - evald(x0)*Fx(x0)
        x1[1] = x0[1] - evald(x0)*Fy(x0)
        x1[2] = x0[2] - evald(x0)*Fz(x0)
        x = np.append(x, x1)
        if(norm(x0-x1) < tol):
            break
        x0 = x1
        i = i +1
    print("Root:", x1)
    print("F(root):", evalF(x1))
    print("Iterations:", i)

    er = np.zeros(i)
    it = 0
    while(it < i):
        er[it] = norm(x[it]) - norm(x1)
        it = it + 1

    plt.plot(np.log(abs(er[:-1])), np.log(abs(er[1:])))
    plt.axis('scaled')
    plt.title("|x_k+1 − α| vs |x_k − α|")
    plt.show()
def evalF(x):
    F  = x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16
    return F

def Fx(x):
    return 2*x[0]

def Fy(x):
    return 8*x[1]

def Fz(x):
    return 8*x[2]


def evald(x):
    d = evalF(x)/(Fx(x)**2 + Fy(x)**2 + Fz(x)**2)
    return d


driver()

