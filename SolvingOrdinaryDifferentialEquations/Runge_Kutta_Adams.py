# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 9:16 下午
# @Author  : Leo
# @FileName: Runge_Kutta_Adams.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com
import numpy as np



class RungeKutta_AdamsSolvingOrdinaryDifferentialEquations:
    """
    用龙格库塔与亚当姆斯方法求解常微分方程，并做对比分析
    """

    def __init__(self, equs_fxy, sol_interval, y0, h_step):
        self.equs_fxy = equs_fxy  # 需要求解的微分方程，y'=f(x,y),f的格式：
        '''
        def f(x,y):
            ...
            return dy
        '''
        if len(sol_interval) == 2:
            self.a, self.b = sol_interval[0], sol_interval[1]  # 求解区间，a，b分别为起始值和终止值
        else:
            raise ValueError("求解区间参数设置不规范，应为[a, b].")
        self.y0 = y0  # 初值,起始条件，y0=y(0)
        self.h = h_step  # 求解步长（区间[a,b]n等分）
        self.n = round(abs(self.b - self.a) / self.h)  # 节点
        self.true_value = None  # 方程的真值

    def _Ord4Runge_Kutta(self):  # 四阶古典Ｒｕｎｇｅ—Ｋｕｔｔａ公式
        y = np.zeros((self.n, 1))
        x = np.zeros((self.n, 1))

        y[0] = self.y0
        x[0] = self.a
        K1, K2, K3, K4 = 0, 0, 0, 0  # 初始化K
        print("%4s %9s %8s %10s"%("x_n", "4阶龙格-库塔", "精确解", "误差"))
        for i in range(1, self.n, 1):
            x[i] = self.a + i * self.h
            K1 = self.equs_fxy(x[i - 1], y[i - 1])
            K2 = self.equs_fxy(x[i - 1] + 1 / 2 * self.h, y[i - 1] + 1 / 2 * self.h * K1)
            K3 = self.equs_fxy(x[i - 1] + 1 / 2 * self.h, y[i - 1] + 1 / 2 * self.h * K2)
            K4 = self.equs_fxy(x[i - 1] + self.h, y[i - 1] + self.h * K3)
            y[i] = y[i - 1] + self.h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
            y_t = x[i] * (3 + x[i] ** 2) / (3 * (1 + x[i] ** 2))
            print(x[i], y[i], y_t, abs(y[i] - y_t))
        return x, y

    def _Ord4Kutta(self):  # 四阶Ｋｕｔｔａ公式
        y = np.zeros((4, 1))
        x = np.zeros((4, 1))

        y[0] = self.y0
        x[0] = self.a
        K1, K2, K3, K4 = 0, 0, 0, 0
        for i in range(1, 4, 1):
            x[i] = self.a + i * self.h * 4
            K1 = self.equs_fxy(x[i - 1], y[i - 1])
            K2 = self.equs_fxy(x[i - 1] + 1 / 3 * self.h * 4, y[i - 1] + 1 / 3 * self.h * 4 * K1)
            K3 = self.equs_fxy(x[i - 1] + 2 / 3 * self.h * 4, y[i - 1] - 1 / 3 * self.h * 4 * K1 + self.h * 4 * K2)
            K4 = self.equs_fxy(x[i - 1] + self.h * 4, y[i - 1] + self.h * 4 * K1 - self.h * 4 * K2 + self.h * 4 * K3)
            y[i] = y[i - 1] + self.h * 4 / 8 * (K1 + 3 * K2 + 3 * K3 + K4)
        return x, y

    def _Ord4ModifiedAdams(self):  # 预测-校正法：四阶Adams预测-校正法
        y = np.zeros((self.n, 1))
        x = np.zeros((self.n, 1))

        y[0] = self.y0
        x[0] = self.a
        temp = 0

        if self.n > 4:
            x1, y1 = self._Ord4Kutta()
            y[0:4] = y1
            print("%4s %6s %6s %10s"%("x_n", "亚当姆斯预测校正", "精确解", "误差"))
            for i in range(4, self.n, 1):
                x[i] = self.a + i * self.h
                temp = y[i] + self.h / 24 * (55 * self.equs_fxy(x[i - 1], y[i - 1]) - 59 * self.equs_fxy(x[i - 2], y[
                    i - 2]) + 37 * self.equs_fxy(x[i - 3], y[i - 3]) - 9 * self.equs_fxy(x[i - 4], y[i - 4]))
                y[i] = y[i - 1] + self.h / 24 * (
                        9 * temp + 19 * self.equs_fxy(x[i - 1], y[i - 1]) - 5 * self.equs_fxy(x[i - 2], y[
                    i - 2]) + self.equs_fxy(x[i - 3], y[i - 3]))
                y_t = x[i] * (3 + x[i] ** 2) / (3 * (1 + x[i] ** 2))
                print(x[i], y[i], y_t, abs(y[i] - y_t))
        else:
            raise ValueError("四阶Adams预测-校正法节点数n必须大于4！")
        return x, y
