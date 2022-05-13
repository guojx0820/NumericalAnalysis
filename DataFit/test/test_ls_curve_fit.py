# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 8:08 下午
# @Author  : Leo
# @FileName: test_ls_curve_fit.py
# @Software: PyCharm
# @Blog    ：https://guojx0820.github.io
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from DataFit.utils.least_square_curve_fit import LeastSquarePolynomialCurveFitting

if __name__ == "__main__":
    # x = np.linspace(0, 5, 20)
    x = np.asarray(
        [1.0, 1.3, 1.5, 1.6, 1.75, 1.8, 1.85, 1.9, 2.0, 2.1,
         2.3, 2.5, 2.6, 2.75, 2.8, 3.0, 3.1, 3.3, 3.45, 3.6])
    y = np.asarray(
        [49.7861, 38.5901, 32.1196, 29.1902, 25.1376, 23.8741, 22.6384,
         21.4692, 19.2307, 17.1448, 13.4711, 10.3518, 8.9906, 7.1734,
         6.6341, 4.7231, 3.9277, 2.6321, 1.8508, 1.2647])
    ls = LeastSquarePolynomialCurveFitting(x, y, k=3)
    ls.fitLeastSquareCurve()
    ls.plt_curve_fit()
