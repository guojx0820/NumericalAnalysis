# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 4:05 下午
# @Author  : Leo
# @FileName: composit_2points_gauss.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

from sympy import *
import numpy as np
import math


def f(t):
    f = exp(t) * cos(t)
    return f


a = 0
b = pi
x = symbols('x')
truth = integrate(f(x), (x, a, b)).evalf()
print(truth)  # 真值

n = 1000  # 步长,就是将(a,b)区间分为多少个块
h = (b - a) / n
result = 0
for k in range(n - 1):
    result += f(a + (k + 1 / 2) * h)
result *= h
# result = h * f(h * (-1 / sqrt(3)) + (b + a) / n) + h * f(h * (1 / sqrt(3)) + (b + a) / n)
print(result.evalf())
