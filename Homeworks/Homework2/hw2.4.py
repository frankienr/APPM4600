import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, np.pi, 30)
y = np.cos(t)
S = 0

for k in range(1, 30):
    S = S + t[k]*y[k]


print('the sum is: ', S)


theta = np.linspace(0, 2*np.pi, 400)
R = 1.2
dr = 0.1
f = 15
p = 0
X = lambda x: R*(1+dr*np.sin(f*x+p))*np.cos(x)
Y = lambda x: R*(1+dr*np.sin(f*x+p))*np.sin(x)

Xt = X(theta)
Yt = Y(theta)

plt.plot(Xt, Yt)
plt.axis('scaled')
plt.show()


for i in range(1, 10):
    R = i
    dr = 0.5
    f = 2 + i
    p = np.random.uniform(0,2)
    X = lambda x: R*(1+dr*np.sin(f*x+p))*np.cos(x)
    Y = lambda x: R*(1+dr*np.sin(f*x+p))*np.sin(x)
    Xt = X(theta)
    Yt = Y(theta)
    plt.plot(Xt, Yt)

plt.axis('scaled')
plt.show()

