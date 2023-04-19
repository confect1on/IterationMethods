import math
import random

import numpy as np
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
n = 5
a = np.array([[max(i, j) for i in range(n)] for j in range(n)], dtype="float64")
y = np.ones((n, 1))
y_second = np.ones((n, 1))
test = np.ones((1, n))
test_two = np.zeros((1,n))
print(y)
eps = 1e-6
prev_eigen_value = random.random()
cond = True
print(a * y)
print(a.T)
while cond:
    y_next = a @ y
    y_second_next = a.T @ y_second
    t = np.inner(y_next.T, y_second_next.T)
    t_2 = np.inner(y.T, y_second_next.T)
    cur_eigen_value = t / t_2
    if abs(cur_eigen_value - prev_eigen_value) < eps:
        cond = False
    prev_eigen_value = cur_eigen_value
    y = y_next
    y_second = y_second_next

print(np.linalg.eigvals(a))
print(prev_eigen_value)
print(a.T)
def jacobi(a : np.array, eps : float=1e-6):
    assert np.allclose(a, a.T)
    A = a.copy()
    n, _ = A.shape
    while True:
        i_max, j_max = 1, 0
        max_abs_element = abs(A[1][0])
        for i in range(n):
            for j in range(n):
                cur_element = abs(A[i][j])
                if i != j and cur_element > max_abs_element:
                    i_max, j_max, max_abs_element = i, j, cur_element
        if np.sum(np.abs(A)) - np.trace(np.abs(A))  < eps:
            return A
        assert np.nonzero(A[i_max][i_max] - A[j_max][j_max])
        p = 2 * max_abs_element / (A[i_max][i_max] - A[j_max][j_max])
        phi = 0.5 * np.arctan(p)
        Q = np.eye(n, n)
        Q[i_max, i_max] = np.cos(phi)
        Q[i_max, j_max] = -np.sin(phi)
        Q[j_max, i_max] = np.sin(phi)
        Q[j_max, j_max] = np.cos(phi)
        first_mul = np.matmul(Q.T, A)
        A = np.matmul(first_mul, Q)
print("Jacobi:")
print(jacobi(a))