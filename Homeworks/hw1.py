import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(1.920, 2.080, 160)
Yx1 = lambda x: (x**9)-18*(x**8)+144*(x**7)-672*(x**6)+2016*(x**5)-4032*(x**4)+5376*(x**3)-4608*(x**2)+2304*x-512
Yx2 = lambda x: (x-2)**9
Ya = Yx1(X)
Yb = Yx2(X)
plt.plot(X, Ya)
plt.plot(X, Yb)
plt.show()