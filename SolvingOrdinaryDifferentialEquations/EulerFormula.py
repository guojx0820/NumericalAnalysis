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


class EulerFormulaSolvingOrdinaryDifferentialEquations:
    """
    用欧拉公式求解常微分方程，包括显式与隐式
    """

    def __init__(self, h_step, n=40, y0, equs_fun, a, b):
        self.equs_fun = equs_fun  # 需要求解的微分方程，符号定义
        if h_step > 0:
            self.h_step = h_step  # 步长
        else:
            raise ValueError("步长设置应为大于0的数！")
        self.y0 = y0  # 初值
        self.n = n  # 方程的阶数
        self.a, self.b = a, b  # 方程
        self.true_value = None  # 方程的真值

