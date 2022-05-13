# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 9:31 下午
# @Author  : Leo
# @FileName: square_root_decomposition.py
# @Software: PyCharm
# @Blog    ：https://guojx0820.github.io
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

import numpy as np


class SquareRootDecompositionAlgorithm:
    """
    平方根分解法解方程组：cholesky分解法和改进的平方根分解法
    """

    def __init__(self, A, b, sol_method="improved"):
        self.A = np.asarray(A, dtype=np.float_)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("系数矩阵不是方阵，行列不相等，不能用高斯消元法求解！")
        else:
            self.n = self.A.shape[0]  # 取方阵的维度
        self._check_symmetric_positive_definite_matrix_()  # 对称正定矩阵判断
        self.b = np.asarray(b, dtype=np.float_)
        if len(self.b) != self.n:
            raise ValueError("右端常数向量维度与系数矩阵维度不匹配！")
        # 平方根分解法的类型：平方根法cholesky和改进的平方根法improve
        self.sol_method = sol_method
        self.x, self.y = None, None  # 线性方程组的解
        self.eps = None  # 验证精度
        self.L, self.D = None, None  # A=LL T或者A=LDL T

    def _check_symmetric_positive_definite_matrix_(self):
        """
        对称正交矩阵判断
        :return:
        """
        if (self.A == self.A.T).all():  # 对称
            if self.A[0, 0] > 0:
                for i in range(1, self.n):  # 正定
                    if np.linalg.det(self.A[i:, i:]) <= 0:
                        raise ValueError("系数矩阵为非正定矩阵！")
            else:
                raise ValueError("系数矩阵为非正定矩阵！")
        else:
            raise ValueError("系数矩阵为非对称矩阵！")

    def fit_solve(self):
        """
        cholesky分解和改进的平方根分解
        :return:
        """
        self.L, self.D = np.eye(self.n), np.zeros((self.n, self.n))
        self.y, self.x = np.zeros(self.n), np.zeros(self.n)
        if self.sol_method == "cholesky":  # 主元
            self.x = self._solve_cholesky_()
        elif self.sol_method == "improved":  # 改进平方根
            self.x = self._solve_improved_cholesky_()
        else:
            raise ValueError("仅适合Doolittle LU分解法和选主元LU分解法！")
        return self.x

    def _solve_cholesky(self):
        """
        平方根法，即cholesky分解法
        :return:
        """
        # 1、公式分解
        for j in range(self.n):
            self.L[j, j] = np.sqrt(self.A[j, j] - sum(self.L[j, :j] ** 2))  # 对角线元素
            for i in range(j + 1, self.n):
                self.L[i, j] = (self.A[i, j] - np.dot(self.L[i, :j], self.L[j, :j])) / self.L[j, j]
        # 2、两次回代求解
        for i in range(self.n):
            self.y[i] = (self.b[i] - np.dot(self.L[i, :i], self.y[:i])) / self.L[i, i]
        for i in range(self.n - 1, -1, -1):
            self.x[i] = (self.y[i] - np.dot(self.L[i:, i], self.x[i:])) / self.L[i, i]
        # 3、验证精度
        self.eps = np.dot(self.A, self.x) - self.b
        return self.x

    def _solve_improved_cholesky_(self):
        """
        改进的平方根分解法
        :return:
        """
        # 1、求解下三角矩阵L和对角矩阵D
        self.D[0, 0] = self.A[0, 0]
        t = np.zeros((self.n, self.n))
        for i in range(1, self.n):
            for j in range(i):
                t[i, j] = self.A[i, j] - np.dot(t[i, :j], self.L[j, :j])
                self.L[i, j] = t[i, j] / self.D[j, j]
            self.D[i, i] = self.A[i, i] - np.dot(t[i, :i], self.L[i, :i])
        # 2、两次回代求解
        for i in range(self.n):
            self.y[i] = self.b[i] - np.dot(self.L[i, :i], self.y[:i])
        for i in range(self.n - 1, -1, -1):
            self.x[i] = self.y[i] / self.D[i, i] - np.dot(self.L[i:, i], self.x[i:])
        # 3、验证精度
        self.eps = np.dot(self.A, self.x) - self.b
        return self.x
