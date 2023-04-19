import math
import numpy as np
import matplotlib.pyplot as plt

from non_linear_equations import draw_function


def lagrange_polynomial(x, n):
    x_points = np.linspace(-1, 1, n)
    print(f"n={n}: {x_points}")
    y_points = f(x_points)
    n = len(x_points)
    accumulate_sum = 0
    for k in range(0, n):
        accumulate_product = 1
        for j in range(0, k):
            accumulate_product *= (x - x_points[j]) / (x_points[k] - x_points[j])
        for j in range(k + 1, n):
            accumulate_product *= (x - x_points[j]) / (x_points[k] - x_points[j])
        accumulate_sum += accumulate_product * y_points[k]
    return accumulate_sum


def f(x):
    return 1 / (1 + 25 * x ** 2)


left = -1
right = 1
x = np.linspace(left, right, 100)
y = f(x)
lagrange_y_20 = lagrange_polynomial(x, 20)
lagrange_y_5 = lagrange_polynomial(x, 5)
plt.plot(x, y, color='r', label="f(x)")
plt.plot(x, lagrange_y_20, color='g', label="lagrange(x) as n = 20")
plt.plot(x, lagrange_y_5, color='b', label="lagrange(x) as n = 5")
plt.legend()
plt.show()

print(f(-1))
print(lagrange_polynomial(-1, 5))
print(lagrange_polynomial(-1, 20))
