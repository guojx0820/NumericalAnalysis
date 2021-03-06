# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 4:57 下午
# @Author  : Leo
# @FileName: test_gauss_legendre.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

import numpy as np
import sympy
import matplotlib.pyplot as plt
from NumericalIntegration.gauss_legendre_int import GaussLegendreIntegration


def fun(x):
    """
    定义积分的被积函数
    :param x:自变量
    :return:被积函数
    """
    return np.cos(x) * np.exp(x)


if __name__ == "__main__":
    gauss_zeros_num = np.arange(8, 20, 1)
    int_accurate = -0.5 * (np.exp(np.pi) + 1)
    print("精确值:", int_accurate)
    precision = []
    for num in gauss_zeros_num:
        legendre = GaussLegendreIntegration(fun, [0, np.pi], zeros_num=num)
        int_value = legendre.cal_int()
        precision.append(int_accurate - int_value)
        print("num:%d,积分值:%.15f,误差:%.15e" % (num, int_value, precision[-1]))  # append list中最后一个值用precision[-1]
    plt.figure(figsize=(8, 6))
    plt.title("Gauss-Legendre Quadrature Formula Error Curve", fontdict={"fontsize": 15})
    plt.xlabel("The Number of Gauss Zero-Points", fontdict={"fontsize": 12})
    plt.ylabel("Error", fontdict={"fontsize": 12})
    plt.plot(gauss_zeros_num, precision, 'ro-')
    plt.grid(axis='y', color='b', linestyle='--', linewidth=0.5)
    plt.savefig("/Users/leo/Desktop/Gauss-Legendre.png", dpi=300, bbox_inches="tight")
    plt.show()
    # legendre._cal_gauss_zeros_points_()
