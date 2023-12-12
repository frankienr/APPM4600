import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from scipy.linalg import lu_factor, lu_solve
import time

def driver():
    ''' create matrix for testing different ways of solving a square
    linear system'''
    '''' N = size of system'''
    N = 100
    ''' Right hand side'''
    b = np.random.rand(N,1)
    A = np.random.rand(N,N)
    #x = scila.solve(A,b)
    #test = np.matmul(A,x)
    t = time.time()
    for i in range(11):
        (lu, piv) = lu_factor(A)
    t = (time.time()-t)
    print("time to create LU:", t/1)
    t = time.time()
    for i in range(1):
        x = lu_solve((lu, piv), b)
    t = (time.time()-t)
    print("time to solve with LU:", t/1)
    test = A@x
    r = la.norm(test-b)
    print(r)
    ''' Create an ill-conditioned rectangular matrix '''
    N = 10
    M = 5
    A = create_rect(N,M)
    b = np.random.rand(N,1)
    print("Solving with QR")
    Q,R = scila.qr(A)
    Qb = np.dot(Q.T, b)
    v_qr = scila.lstsq(R,Qb)[0]
    test_qr = np.matmul(A, v_qr)
    r_qr = la.norm(test_qr-b)
    print(r_qr)

    print("Solving with normal equation")
    AtA = A.T @ A
    Atb = A.T @ b
    x = scila.solve(AtA, Atb)
    test = la.norm(A@x-b)
    print(test)
def create_rect(N,M):
    ''' this subroutine creates an ill-conditioned rectangular matrix'''
    a = np.linspace(10,20,M)
    d = 10**(-a)
    D2 = np.zeros((N,M))
    for j in range(0,M):
        D2[j,j] = d[j]
    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(N,N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1,R)
    A = np.random.rand(M,M)
    Q2,R = la.qr(A)
    test = np.matmul(Q2,R)
    B = np.matmul(Q1,D2)
    B = np.matmul(B,Q2)
    return B
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
