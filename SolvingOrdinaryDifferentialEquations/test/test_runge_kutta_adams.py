# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 11:02 下午
# @Author  : Leo
# @FileName: test_runge_kutta_adams.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

import matplotlib.pyplot as plt
from SolvingOrdinaryDifferentialEquations.Runge_Kutta_Adams import RungeKutta_AdamsSolvingOrdinaryDifferentialEquations


def equs_fxy(x, y):  # 预测值
    return 1 - 2 * x * y / (1 + x ** 2)


if __name__ == "__main__":
    h = 0.1
    y0 = 0
    sol_interval = [0, 2]
    runge_kutta_adams = RungeKutta_AdamsSolvingOrdinaryDifferentialEquations(equs_fxy, sol_interval, y0, h)
    x, y = runge_kutta_adams._Ord4Runge_Kutta()
    t, u = runge_kutta_adams._Ord4ModifiedAdams()
    y_t = x * (3 + x ** 2) / (3 * (1 + x ** 2))
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Runge_Kutta_Solution-' + str(h), color="r")
    plt.scatter(t, u, label='Adams_Solution-' + str(h), color="g")
    plt.plot(x, y_t, label='True_Solution-' + str(h), color="b")
    plt.title("Runge-Kutta & Adams Formula Solving Differential Equations")
    plt.legend()
    plt.savefig("/Users/leo/Desktop/runge_kutta_adams_01.png", dpi=300, bbox_inches="tight")
    plt.figure(figsize=(8, 6))
    plt.plot(x, abs(y - y_t), label='Runge_Kutta_Error-' + str(h), color="r")
    plt.plot(x, abs(u - y_t), label='Adams_Error-' + str(h), color="g")
    plt.title("Error Curve of Runge-Kutta & Adams Formula Solving Differential Equations ")
    plt.legend()
    plt.savefig("/Users/leo/Desktop/error_runge_kutta_adams_01.png", dpi=300, bbox_inches="tight")
    plt.show()
