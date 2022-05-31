# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 7:40 下午
# @Author  : Leo
# @FileName: EulerFormula.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com


import numpy as np
import matplotlib.pyplot as plt


def funEval(x, y):  # 近似值
    fxy = 1 - 2 * x * y / (1 + x ** 2)  # 1
    # fxy=2*y/x+x**2*np.e**x #2
    # stablity
    # fxy=-30*y
    # fxy= np.e**(-x**2)
    # fxy=1-y
    # fxy=2*y/x+x**2*np.e**x
    return fxy


def funtrue(x):  # 真实值
    ft = x * (3 + x ** 2) / (3 * (1 + x ** 2))  # 1
    # ft=x**2*(np.e**x-np.e) #2
    # stablityu
    # ft = np.e**(-30*x)
    # ft = 1-np.e**(-x)
    # ft=x**2*(np.e**x-np.e)
    return ft


def Euler(a, b, f, y0, n):  # Ｅｕｌｅｒ公式
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    for i in range(1, n, 1):
        x[i] = a + i * h
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
    return x, y


def ModEuler(a, b, f, y0, n):  # 改进Ｅｕｌｅｒ公式
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    for i in range(1, n, 1):
        x[i] = a + i * h
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
        y[i] = y[i - 1] + h / 2 * (f(x[i - 1], y[i - 1]) + f(x[i], y[i]))
    return x, y


def Heun(a, b, f, y0, n):  # 二阶Ｒｕｎｇｅ—Ｋｕｔｔａ方法：Ｈｅｕｎ公式
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    K1, K2 = 0, 0
    for i in range(1, n, 1):
        x[i] = a + i * h
        K1 = f(x[i - 1], y[i - 1])
        K2 = f(x[i - 1] + 2 / 3 * h, y[i - 1] + 2 / 3 * h * K1)
        y[i] = y[i - 1] + h / 4 * (K1 + 3 * K2)
    return x, y


def Ord3Kutta(a, b, f, y0, n):  # 三阶Kutta公式
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    K1, K2, K3 = 0, 0, 0
    for i in range(1, n, 1):
        x[i] = a + i * h
        K1 = f(x[i - 1], y[i - 1])
        K2 = f(x[i - 1] + 1 / 2 * h, y[i - 1] + 1 / 2 * h * K1)
        K3 = f(x[i - 1] + h, y[i - 1] - h * K1 + 2 * h * K2)
        y[i] = y[i - 1] + h / 6 * (K1 + 4 * K2 + K3)
    return x, y


def Ord3Heun(a, b, f, y0, n):  # 三阶Ｈｅｕｎ公式
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    K1, K2, K3 = 0, 0, 0
    for i in range(1, n, 1):
        x[i] = a + i * h
        K1 = f(x[i - 1], y[i - 1])
        K2 = f(x[i - 1] + 1 / 3 * h, y[i - 1] + 1 / 3 * h * K1)
        K3 = f(x[i - 1] + 2 / 3 * h, y[i - 1] + 2 / 3 * h * K2)
        y[i] = y[i - 1] + h / 4 * (K1 + 3 * K2)
    return x, y


def Ord4Kutta(a, b, f, y0, n):  # 四阶古典Ｒｕｎｇｅ—Ｋｕｔｔａ公式
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    K1, K2, K3, K4 = 0, 0, 0, 0
    for i in range(1, n, 1):
        x[i] = a + i * h
        K1 = f(x[i - 1], y[i - 1])
        K2 = f(x[i - 1] + 1 / 2 * h, y[i - 1] + 1 / 2 * h * K1)
        K3 = f(x[i - 1] + 1 / 2 * h, y[i - 1] + 1 / 2 * h * K2)
        K4 = f(x[i - 1] + h, y[i - 1] + h * K3)
        y[i] = y[i - 1] + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return x, y


def Ord4Kutta2(a, b, f, y0, n):  # 四阶Ｋｕｔｔａ公式
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    K1, K2, K3, K4 = 0, 0, 0, 0
    for i in range(1, n, 1):
        x[i] = a + i * h
        K1 = f(x[i - 1], y[i - 1])
        K2 = f(x[i - 1] + 1 / 3 * h, y[i - 1] + 1 / 3 * h * K1)
        K3 = f(x[i - 1] + 2 / 3 * h, y[i - 1] - 1 / 3 * h * K1 + h * K2)
        K4 = f(x[i - 1] + h, y[i - 1] + h * K1 - h * K2 + h * K3)
        y[i] = y[i - 1] + h / 8 * (K1 + 3 * K2 + 3 * K3 + K4)
    return x, y


def ExpAdamsK1(a, b, f, y0, n):  # 二步显式Adams
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    for i in range(1, n, 1):
        x[i] = a + i * h
        if i == 1:
            y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
        else:
            y[i] = y[i - 1] + h / 2 * (3 * f(x[i - 1], y[i - 1]) - f(x[i - 2], y[i - 2]))
    return x, y


def ExpAdamsK2(a, b, f, y0, n):  # 三步显式Adams
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    if n >= 2:
        for i in range(1, n, 1):
            x[i] = a + i * h
            if i <= 2:
                y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
            else:
                y[i] = y[i - 1] + h / 12 * (
                        23 * f(x[i - 1], y[i - 1]) - 16 * f(x[i - 2], y[i - 2]) + 5 * f(x[i - 3], y[i - 3]))
    else:
        print("n must be larger than 2")
    return x, y


def ModifiedAdamsK2(a, b, f, y0, n):  # 预测-校正法：四阶Adams预测-校正法
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    y[0] = y0
    x[0] = a
    temp = 0

    if n > 4:
        x1, y1 = Ord4Kutta2(a, h * 4, f, y0, 4)
        y[0:4] = y1
        for i in range(4, n, 1):
            x[i] = a + i * h
            temp = y[i] + h / 24 * (
                    55 * f(x[i - 1], y[i - 1]) - 59 * f(x[i - 2], y[i - 2]) + 37 * f(x[i - 3], y[i - 3]) - 9 * f(
                x[i - 4], y[i - 4]))
            y[i] = y[i - 1] + h / 24 * (
                    9 * temp + 19 * f(x[i - 1], y[i - 1]) - 5 * f(x[i - 2], y[i - 2]) + f(x[i - 3], y[i - 3]))
    else:
        print("n must be larger than 4")

    return x, y


##y'=-30y,y(0)=1, 0<=x<=0.6
def ImplictEuler(a, b, n):  # 显式Euler公式和隐式Euler法精确度的比较
    h = np.abs(b - a) / (n - 1)
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))
    y0 = 1
    y[0] = y0
    x[0] = a
    K1, K2, K3, K4 = 0, 0, 0, 0
    for i in range(1, n, 1):
        x[i] = a + i * h
        y[i] = y[i - 1] / (1 + 30 * h)
    yt = np.e ** (-30 * x)
    return x, y, yt


def main():
    a, b = 0, 2  # 12

    n = [4, 5, 10, 20, 40, 80, 160, 320, 640]
    # n=[5]
    y0 = 0  # 1
    # y0 = 0 #2

    # stablity
    # a,b = 1,2
    # y0 = 0
    # for i in n:
    #     x,y=Euler(a,b,funEval,y0,i)
    #     plt.plot(x,y,label='Euler-solution-'+str(i))
    #     plt.plot(x,funtrue(x),label='True-solution-'+str(i))
    #     plt.legend()
    #     plt.show()

    for i in n:
        # x,y=ModEuler(a,b,funEval,y0,i)
        # x,y=ModifiedAdamsK2(a,b,funEval,y0,i)
        # x,y,yt=ImplictEuler(0,0.6,i)
        x, y = ModifiedAdamsK2(a, b, funEval, y0, i)
        yt = funtrue(x)
        plt.plot(x, y, label='Euler-solution-' + str(i))
        plt.plot(x, yt, label='True-solution-' + str(i))
        plt.legend()
        plt.show()
        print(x[0:5], (y - yt)[0:5])
        print(y)


if __name__ == '__main__':
    main()
