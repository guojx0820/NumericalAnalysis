# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 5:51 下午
# @Author  : Leo
# @FileName: least_square_curve_fit.py
# @Software: PyCharm
# @Blog    ：https://guojx0820.github.io
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

import numpy as np
import sympy
import matplotlib.pyplot as plt
from DataFit.utils.square_root_decomposition import SquareRootDecompositionAlgorithm


class LeastSquarePolynomialCurveFitting:
    """
    多项式曲线（线型）拟合，k代表拟合的阶次
    """

    def __init__(self, x, y, k=3, w=None):
        """
        初始化函数，用于初始化参数与模型
        :param x:自变量
        :param y:因变量
        :param k:曲线多项式拟合的阶次
        :param w:拟合权重
        """
        self.x, self.y = np.asarray(x, dtype=np.float_), np.asarray(y, dtype=np.float_)
        self.k = k  # 多项式曲线拟合的最高阶次
        if len(self.x) != len(self.y):
            raise ValueError("离散数据点长度不一致！")  # 离散数据点x不等于y，触发异常
        else:
            self.n = len(self.x)  # 统计离散数据点的个数
        if w is None:
            self.w = np.ones(self.n)  # 所有权重默认情况下一致，均为1
        else:
            if len(self.w) != self.n:
                raise ValueError("权重长度与离散数据点的长度不一致！")  # 权重长度与离散数据点长度不统一时触发异常
            else:
                self.w = np.asarray(w, dtype=np.float_)
        self.fit_poly = None  # 定义曲线拟合多项式
        self.poly_coefficient = None  # 多项式的系数向量
        self.polynomial_orders = None  # 多项式拟合系数的阶次
        self.fit_error = None  # 多项式拟合误差向量
        self.mse = np.infty  # 多项式拟合均方根误差

    def fitLeastSquareCurve(self):
        """
        最小二乘法拟合多项式曲线
        :return:
        """
        c = np.zeros(2 * self.k + 1)  # 初始化系数矩阵的不同元素，均设置为0,这仅仅是储存数据的一维数组
        b = np.zeros(self.k + 1)  # 右端常数向量
        for k in range(2 * self.k + 1):
            c[k] = np.dot(self.w, np.power(self.x, k))  # 利用循环构造并计算系数矩阵
        for k in range(self.k + 1):
            b[k] = np.dot(self.w, self.y * np.power(self.x, k))  # 利用循环构造并计算右端常数向量
        C = np.zeros((self.k + 1, self.k + 1))  # 构造并初始化（n+1）*（n+1）的矩阵,此为全零矩阵（初始化完成）
        for k in range(self.k + 1):
            C[k, :] = c[k:self.k + k + 1]  # 利用循环对初始化后的系数矩阵赋值，方法是将计算完成的所有系数放在一个一维数组中，利用数组，逐行对系数矩阵赋值
        print("系数集合：", c, "系数矩阵：", C, sep='\n')

        # 采用平方根分解法求解线性方程组的解
        srd = SquareRootDecompositionAlgorithm(C, b)  # 类实例化对象并传递参数
        srd.fit_solve()  # 对象调用方法
        self.poly_coefficient = srd.x  # 计算多项式系数并保留三位有效数字

        t = sympy.Symbol("t")
        self.fit_poly = self.poly_coefficient[0] * 1
        for p in range(1, self.k + 1):
            px = np.power(t, p)  # 幂次
            self.fit_poly += self.poly_coefficient[p] * px
        poly = sympy.Poly(self.fit_poly, t)
        self.polynomial_orders = poly.monoms()[::-1]  # 阶次
        print("多项式系数：", self.poly_coefficient, "拟合多项式：", self.fit_poly, "多项式阶次：", self.polynomial_orders, sep='\n')

        self.cal_fit_error()  # 误差分析

    def cal_fit_error(self):
        """
        计算拟合的误差和均方根误差
        :return:
        """
        y_fit = self.cal_x0(self.x)
        self.fit_error = self.y - y_fit  # 误差向量
        self.mse = np.sqrt(np.mean(self.fit_error ** 2))

    def cal_x0(self, x0):
        """
        求解给定点的拟合值
        :param x0:
        :return:
        """
        t = self.fit_poly.free_symbols.pop()
        fit_poly = sympy.lambdify(t, self.fit_poly)  # 符号转化，将符号化公式转化为可以运算的公式
        return fit_poly(x0)  # 返回x0的多项式的值

    def plt_curve_fit(self):
        """
        拟合曲线及离散数据点的可视化
        :return:
        """
        xi = np.linspace(min(self.x), max(self.x), 100)
        yi = self.cal_x0(xi)  # 计算拟合值
        plt.figure(figsize=(8, 6))
        plt.plot(xi, yi, "k-", lw=1.5, label="Fitting Curve")
        plt.plot(self.x, self.y, "ro", label="Original data")
        plt.legend(fontsize=12)
        plt.xlabel("X", fontdict={"fontsize": 14})
        plt.ylabel("Y", fontdict={"fontsize": 14})
        plt.title("The least aquare fitted curve and original data", fontdict={"fontsize": 16})
        plt.text(3, 42, "MSE=%.2e" % self.mse, fontsize=12,
                 bbox=dict(facecolor="w", alpha=0.3, boxstyle="round"))
        coe_plt = [round(i, 3) for i in self.poly_coefficient]
        plt.text(1, 3, "Y = -" + str(abs(coe_plt[3])) + "$X^{3}$ + " + str(coe_plt[2]) + "$X^{2}$ - "
                 + str(abs(coe_plt[1])) + "X + " + str(coe_plt[0]), fontsize=12,
                 bbox=dict(facecolor="w", alpha=0.3, boxstyle="round"))
        plt.savefig("/Users/leo/Desktop/LeastAquareFit.png", dpi=300, bbox_inches="tight")
        plt.show()
