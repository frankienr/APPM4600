import numpy as np
import math
x=9.999999995000000*10**(-10)
y = (np.e**x)
y = y-1
print('y = ', y)
n = 3
print('relative error (with n =', n-1, ')=', (10**(-9*n)/math.factorial(n))/y)