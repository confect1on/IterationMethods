import math
import numpy as np
import matplotlib.pyplot as plt


def draw_function(func, left, right):
    x = np.linspace(left, right, 100)
    y = func(x)
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)
    ax.set(xlim=(left, right), xticks=np.arange(left, right),
           ylim=(-10, 10), yticks=np.arange(-10, 10))
    plt.axvline(0)
    plt.axhline(0)
    plt.show()


def draw_function_system(func1, func2, x_left, x_right, y_left, y_right):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    x, y = np.meshgrid(
        np.linspace(x_left, x_right, 100),
        np.linspace(y_left, y_right, 100))
    z1 = func1(x, y)
    z2 = func2(x, y)

    ax.plot_surface(x, y, z1)
    ax.plot_surface(x, y, z2)

    plt.show()


def bisection(func, left, right, eps=1e-6):
    f_left = func(left)
    f_right = func(right)
    mid = (left + right) / 2
    assert f_left * f_right < 0
    while abs(func(mid)) >= eps:
        assert np.sign(f_left) != np.sign(f_right)
        mid = (left + right) / 2
        f_mid = func(mid)
        if np.sign(f_mid) != np.sign(f_left):
            right = mid
            f_right = func(right)
        elif np.sign(f_mid) != np.sign(f_right):
            left = mid
            f_left = func(left)
        else:
            assert False
    return mid


def chord_method(func, left, right, eps=1e-6):
    x0 = left
    x1 = right
    while abs(func(x1)) >= eps:
        t = x1
        x1 = x1 - ((x1 - x0) / (func(x1) - func(x0))) * func(x1)
        x0 = t
    return x1


def secant_method_for_system(func1, func2, left, right, eps=1e-6):
    np.random.rand()
    pass


def first_func(x):
    return x ** 3 - 3 * x - 2 * math.e ** (-x)


def second_func(x):
    return x - 1 / np.arctan(x)


def first_equation(x, y):
    return np.sin(x + 1) - y - 1.2


def second_equation(x, y):
    return 2 * x


if __name__ == '__main__':
    draw_function(second_func, -10, 10)
    bisect_arg = bisection(second_func, 1, 3)
    print(f"bisection: x:{bisect_arg}, f(x):{second_func(bisect_arg)}")
    chord_arg = chord_method(second_func, 1, 3)
    print(f"chord: x:{chord_arg}, f(x):{second_func(chord_arg)} ")
