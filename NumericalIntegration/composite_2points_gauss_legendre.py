# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 5:11 下午
# @Author  : Leo
# @FileName: composite_2points_gauss_legendre.py
# @Software: PyCharm
# @Blog    ：https://guojxblog.cn
# @GitHub  ：https://github.com/guojx0820
# @Email   ：guojiaxiang0820@gmail.com

class CompositeGaussLegendreIntegration:
    """
    复化两点高斯——勒让德求积公式，等距节点数自定义
    """

    def __init__(self, int_fun, int_interval, interval_num=4):
        self.int_fun = int_fun  # 被积函数，符号定义
        if len(int_interval) == 2:
            self.a, self.b = int_interval[0], int_interval[1]  # 被积区间
        else:
            raise ValueError("积分区间参数设置不规范，应为[a, b].")
        self.interval_num = interval_num
        self.int_value = None

    def cal_int(self):
        """
        复化两点高斯——勒让德求积公式
        :return:
        """
        fun_value = 0  # 初始化函数值
        interval_len = (self.b - self.a) / self.interval_num
        for k in range(self.interval_num - 1):
            fun_value += self.int_fun(self.a + (k + 1 / 2) * interval_len)
        self.int_value *= interval_len
        return self.int_value.evalf()
