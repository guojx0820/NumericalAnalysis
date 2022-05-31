# -*- coding: utf-8 -*-
# @Time    : 2022/5/14 7:46 下午
# @Author  : Leo
# @FileName: composite_quadrature_formula.py.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com
import numpy as np
import sympy


class CompositeQuadratureItegration:
    """
    复化求积公式，复化梯形公式，复化辛普森公式，复化科特斯公式
    """

    def __init__(self, int_fun, int_interval, interval_num=16, int_type="simpson"):
        self.int_fun = int_fun  # 被积函数，符号定义
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 被积区间
        else:
            raise ValueError("积分区间参数设置不规范，应为[a, b].")
        self.n = int(int_interval)  # 默认等分子区间数为
        self.int_type = int_type  # 积分公式类型，默认采用符合辛普森
        self.int_value = None  # 积分值结果
        self.int_remainder = None  # 积分值余项

    def cal_int(self):
        """
        根据参数设置，选择不同的积分类型
        :return:
        """
        t = self.int_fun.free_symbols.pop()  # 被积函数自由变量
        fun_expr = sympy.lambdify(t, self.int_fun)  # 转化为lambda函数
        if self.int_type == "trapezoid":
            self._cal_trapezoid_(t, fun_expr)
        elif self.int_type == "simpson":
            self._cal_simpson_(t, fun_expr)
        elif self.int_type == "cotes":
            self._cal_cotes_(t, fun_expr)
        else:
            raise ValueError("复化积分类型仅支持trapezoid，simpson，cotes.")

    def _cal_trapezoid_(self, t, fun_expr):
        """
         复化梯形公式
        :param t:自由变量
        :param fun_expr:被积函数
        :return:
        """

    pass

    def _cal_simpson_(self, t, fun_expr):
        """
         复化辛普森公式
        :param t:自由变量
        :param fun_expr:被积函数
        :return:
        """

    pass

    def _cal_cotes_(self, t, fun_expr):
        """
         复化科特斯公式
        :param t:自由变量
        :param fun_expr:被积函数
        :return:
        """

    pass
