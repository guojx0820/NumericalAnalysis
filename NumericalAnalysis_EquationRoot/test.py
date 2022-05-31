import numpy as np
from sympy import *
import matplotlib.pyplot as plt


def f(x):
    return 1 / (1 + x ** 2)


def ChebyshevXGet():
    ans = np.array(list(map(lambda x: np.cos((21 - 2 * x) / 22 * np.pi), range(11))))
    return ans * 5


def draw(L):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.linspace(-5, 5, 100)
    y = f(x)
    Ly = []
    for xx in x:
        Ly.append(L.subs(n, xx))
    plt.plot(x, y, label='原函数')
    plt.plot(x, Ly, label='Lagrange插值函数')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.savefig('1.png')
    plt.show()


def lossCal(L):
    x = np.linspace(-5, 5, 101)
    y = f(x)
    Ly = []
    for xx in x:
        Ly.append(L.subs(n, xx))
    Ly = np.array(Ly)
    temp = Ly - y
    temp = abs(temp)
    print(temp.mean())


if __name__ == '__main__':
    x = ChebyshevXGet()
    y = f(x)

    n, m = symbols('n m')
    init_printing(use_unicode=True)
    L = 0
    for k in range(11):
        temp = y[k]
        for i in range(11):
            if i != k:
                temp *= (n - x[i]) / (x[k] - x[i])
        L += temp
    lossCal(L)
    draw(L)
