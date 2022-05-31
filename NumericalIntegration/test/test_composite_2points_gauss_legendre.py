# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 5:56 下午
# @Author  : Leo
# @FileName: test_composite_2points_gauss_legendre.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import symbol
from NumericalIntegration.composite_2points_gauss_legendre import CompositeGaussLegendreIntegration


def fun(x):
    """
    定义积分被积函数
    :param x: 自由变量
    :return:
    """
    return np.cos(x) * np.exp(x)


if __name__ == "__main__":
    interval_num = np.arange(100000, 1000000, 100000)
    int_accurate = -0.5 * (np.exp(np.pi) + 1)
    print("精确解:", int_accurate)
    precision = []
    for num in interval_num:
        legendre = CompositeGaussLegendreIntegration(fun, [0, np.pi], interval_num=num)
        int_value = legendre.cal_int()
        precision.append(int_accurate - int_value)
        print("num=%d,积分值:%.15f,误差:%.15e" % (num, int_value, precision[-1]))  # append list中最后一个值用precision[-1]
    plt.figure(figsize=(8, 6))
    plt.title("Composite Gauss Quadrature Formula Error Curve", fontdict={"fontsize": 16})
    plt.xlabel("The Number of Integral Intervals", fontdict={"fontsize": 12})
    plt.ylabel("Error", fontdict={"fontsize": 12})
    plt.plot(interval_num, precision, 'ro-')
    plt.grid(axis='y', color='b', linestyle='--', linewidth=0.5)
    plt.savefig("/Users/leo/Desktop/CompositeGauss.png", dpi=300, bbox_inches="tight")
    plt.show()
