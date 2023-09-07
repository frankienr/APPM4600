import numpy as np
import numpy.linalg as la
import math
def driver():
    # PROBLEM 1
    n = 2
    x = np.linspace(1, 2, n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language

    #these will produce the vectors
    f = lambda x: x
    g = lambda x: ((-1)**x)*(1/x)
    y = f(x)
    w = g(x)
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    # print the output
    print('the dot product of', y, ' and ', w,' is : ', dp)

    #PROBLEM 2
    print('Matrix vector multiplication by "hand"')
    # this will represent dong the matrix multiplication manually
    m = np.array([[1,2], [3,4]])
    n = np.array([5,6])
    ans = np.array([])

    print(m, n, '=')
    for i in range(2):
        dot = dotProduct(m[i], n, 2)
        ans = np.append(ans, [dot])
    print(ans)

    m = np.array([[1,2,3,4], [5,6,7,8], [9, 10, 11, 12], [13, 14, 15, 16]])
    n = np.array([1,2,3,4])
    ans = np.array([])

    print(m, n, '=')
    for i in range(4):
        dot = dotProduct(m[i], n, 4)
        ans = np.append(ans, [dot])
    print(ans)

    print('Matrix vector multiplication with numpy')
    #here we use the built in fuction to matrix multiply
    m = np.array([[1,2], [3,4]])
    n = np.array([5,6])
    ans = np.matmul(m,n)
    print(m, n, '=', ans)


    m = np.array([[1,2,3,4], [5,6,7,8], [9, 10, 11, 12], [13, 14, 15, 16]])
    n = np.array([1,2,3,4])
    ans = np.matmul(m,n)
    #we can print matrices out and it will auto make them look nice
    print(m, n, '=', ans)
    return

def dotProduct(x,y,n):
# Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    #starting with 0. so we are not accidentally using integers
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp
driver()