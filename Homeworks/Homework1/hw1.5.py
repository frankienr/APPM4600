import numpy as np
import matplotlib.pyplot as plt


x1 = np.pi
x2 = 10**6

y = np.linspace(-16, 0, 17)
delty = lambda y: 10**y
delta = delty(y)

Y1real = lambda d: np.cos(x1+d) -np.cos(x1)
Y1 = lambda d: -2*np.sin(x1+(0.5*d))*np.sin(0.5*d)
Y1a = Y1(delta)
Y1r = Y1real(delta)


Y2real = lambda d: np.cos(x2+d) -np.cos(x2)
Y2 = lambda d: -2*np.sin(x2+(0.5*d))*np.sin(0.5*d)
Y2a = Y2(delta)
Y2r = Y2real(delta)



Taylor1 = lambda d: -1*d*np.sin(x1) - d**2*0.5*np.cos(x1+0.5*d)
T1 = Taylor1(delta)
Taylor2 = lambda d: -1*d*np.sin(x2) - d**2*0.5*np.cos(x2+0.5*d)
T2 = Taylor2(delta)

#plt.plot(np.log(delta), Y1a)
plt.plot(np.log(delta), Y1r)
plt.plot(np.log(delta), T1)

#plt.plot(np.log(delta), Y2a)
#plt.plot(np.log(delta), Y2r)
#plt.plot(np.log(delta), T2)
plt.title("x = pi, Taylor")

plt.show()

