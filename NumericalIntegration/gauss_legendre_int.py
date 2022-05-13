# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 4:05 下午
# @Author  : Leo
# @FileName: gauss_legendre_int.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

import numpy as np
import sympy
import math


class GaussLegendreIntegration:
    """
    高斯——勒让德求积公式，核心部分求解零点与系数
    """

    def __init__(self, int_fun, int_interval, zeros_num=10):
        self.int_fun = int_fun  # 被积函数，符号定义
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 被积区间
        else:
            raise ValueError("积分区间参数设置不规范，应为[a,b].")
        self.n = int(zeros_num)  # 正交多项式的零点数
        self.zeros_points = None  # 勒让德高斯零点
        self.int_value = None  # 积分值结果
        self.A_k = None  # 求积系数

    def cal_int(self):
        """
        高斯——勒让德求积公式
        :return:
        """
        self._cal_Ak_coef_()  # 求解求积系数Ak与零点
        fun_val = self.int_fun(self.zeros_points)  # 零点函数值
        self.int_value = np.dot(self.A_k, fun_val)  # 插值型求积公式
        return self.int_value

    def _cal_gauss_zeros_points_(self):
        """
        计算高斯零点
        :return:
        """
        t = sympy.Symbol('t')
        # 勒让德多项式构造
        p_n = (t ** 2 - 1) ** self.n / (math.factorial(self.n) * 2 ** self.n)
        diff_p_n = sympy.diff(p_n, t, self.n)  # 多项式的n阶导数
        # print(diff_p_n)
        # 求解高斯——勒让德多项式的全部零点
        self.zeros_points = np.asarray(sympy.solve(diff_p_n, t), dtype=np.float_)
        print("高斯节点:", self.zeros_points, sep="\n")
        return diff_p_n, t

    def _cal_Ak_coef_(self):
        """
        计算Ak系数
        :return:
        """
        diff_p_n, t = self._cal_gauss_zeros_points_()  # 求解高斯零点，传递函数和符号参数
        Ak_poly = sympy.lambdify(t, 2 / ((1 - t ** 2) * (diff_p_n.diff(t, 1) ** 2)))
        self.A_k = Ak_poly(self.zeros_points)  # 求解求积系数Ak
        # 区间转换，[a, b] --> [-1, 1]
        self.A_k = self.A_k * (self.b - self.a) / 2
        self.zeros_points = (self.b - self.a) / 2 * self.zeros_points + (self.a + self.b) / 2
        print("求积系数:", self.A_k, sep="\n")
