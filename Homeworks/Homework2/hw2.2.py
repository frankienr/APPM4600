import numpy as np


A = [[1,1],
     [1+10**(-10), 1-10**(-10)]]

U, S, VT = np.linalg.svd(A)
print(S)

b1 = 1.0*10**(-5)
b2 = 1.0*10**(-5)

dx1 = b1 +(10**10)*(b2-b1)
dx2 = b1 +(10**10)*(b1-b2)

dx = ((dx1**2 + dx2**2)**0.5)
print(dx/2)