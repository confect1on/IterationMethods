import random

import numpy as np

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
n = 5
a = np.array([[max(i, j) for i in range(n)] for j in range(n)], dtype="float64")
test = np.ones((1, n))
test_two = np.zeros((1, n))


def scalar_products(matrix: np.array, eps: float = 1e-6):
    y = np.ones((n, 1))
    y_second = np.ones((n, 1))
    prev_eigen_value = random.random()
    cond = True
    while cond:
        y_next = matrix @ y
        y_second_next = matrix.T @ y_second
        t = np.inner(y_next.T, y_second_next.T)
        t_2 = np.inner(y.T, y_second_next.T)
        cur_eigen_value = t / t_2
        if abs(cur_eigen_value - prev_eigen_value) < eps:
            cond = False
        prev_eigen_value = cur_eigen_value
        y = y_next
        y_second = y_second_next
    return prev_eigen_value


print("Numpy eigen values: ", np.linalg.eigvals(a))
max_eigen_value = scalar_products(a)
print("Max eigen value: ", max_eigen_value)
n, _ = a.shape

b = a - max_eigen_value * np.eye(n, n)
temp_value = scalar_products(b)
min_eigen_value = temp_value + max_eigen_value
print("Min eigen value:", min_eigen_value)
print(a)
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
        if max_abs_element < eps:
            return A
        assert np.nonzero(A[i_max][i_max] - A[j_max][j_max])
        p = (2 * a[i_max, j_max]) / (A[i_max][i_max] - A[j_max][j_max])
        phi = 0.5 * np.arctan(p)
        Q = np.eye(n, n)
        Q[i_max, i_max] = np.cos(phi)
        Q[i_max, j_max] = -np.sin(phi)
        Q[j_max, i_max] = np.sin(phi)
        Q[j_max, j_max] = np.cos(phi)
        print(Q)
        A = Q.T @ A @ Q
print("Jacobi:")
print(jacobi(a))
