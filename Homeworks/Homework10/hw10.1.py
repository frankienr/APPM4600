import numpy as np
import matplotlib.pyplot as plt

def driver():
    mac = lambda x: x - (x**3)/6 + (x**5)/120 - (x**7)/5040 + (x**9)/362880
    pade33 = lambda x: (x-(7/60)*x**3)/(1+(1/20)*x**2)
    pade42 = lambda x: x/(1+(1/6)*x**2+(7/360)*x**4)
    pade24 = lambda x: (x-(7/60)*x**3)/(1+(1/20)*x**2)

    x = np.linspace(0,5,100)
    plt.figure()
    macx = mac(x)
    pade33x = pade33(x)
    pade42x = pade42(x)
    pade24x = pade24(x)
    plt.title("graph of approximations")
    plt.plot(x, macx, label='MacLaurin')
    plt.plot(x, pade33x, label='Pade 3/3')
    plt.plot(x, pade42x, label='Pade 4/2')
    plt.plot(x, pade24x, label='Pade 2/4')
    plt.legend()
    plt.figure()
    plt.title("graph of errors")
    plt.plot(x, abs(pade33x-macx), label='Pade 3/3')
    plt.plot(x, abs(pade42x-macx), label='Pade 4/2')
    plt.plot(x, abs(pade24x-macx), label='Pade 2/4')
    plt.legend()
    plt.show()

driver()