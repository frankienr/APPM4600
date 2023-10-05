import numpy as np
import matplotlib.pyplot as plt
def driver():
    f = lambda x: np.cos(x)
    h = 0.01*2.**(-np.arange(0,10))
    x = np.pi/2
    fFor = lambda h: (f(x+h) - f(x))/h
    fCen = lambda h: (f(x+h) - f(x-h))/(2*h)

    fForx = fFor(h)
    fCenx = fCen(h)

    print(fForx)
    print(fCenx)

    erFor = fForx + 1
    erCen = fCenx + 1
    plt.plot(np.log(abs(erFor[:-1])), np.log(abs(erFor[1:])))
    plt.plot(np.log(abs(erCen[:-1])), np.log(abs(erCen[1:])))
    plt.axis('scaled')
    plt.show()


driver()