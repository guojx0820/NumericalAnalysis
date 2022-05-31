from sympy import *


def fun():
    """
    求解方程的符号定义
    :return:方程
    error:这里不能用math库中的函数，pow()需要调用符号库中的函数，即sympy.pow()
    或者用from sympy import * 来导入sympy中的所有函数，之后直接写pow()函数即
    是默认的sympy中的函数。
    """
    x = symbols('x')  # 符号变量的定义
    # return 2 * exp(-x) * sin(x) + 2 * cos(x) - 0.25
    return pow(x, 3) + 2.0 * pow(x, 2) + 10.0 * x - 20.0  # 返回函数的值


def fixedPoint(x0, eps, maxIter):
    """
    不动点迭代法的作用是，求解非线性方程的根，采用逐步逼近的方法进行计算
    :param x0:迭代初值
    :param eps:误差精度要求
    :param maxIter:最大迭代次数
    :return:返回值为None
    """
    x = symbols('x')
    fh = fun()
    x_n = x0  # 定义初值
    k = 0  # 初始化迭代次数
    errval = 0  # 初始化误差
    print('%3s %10s %18s' % ('迭代次数', '方程近似值', '迭代误差'))
    for k in range(maxIter):
        x_b = x_n  # 代表x_n
        x_n = 20 / (pow(x_b, 2) + 2 * x_b + 10)  # 写出第一种迭代函数格式
        errval = abs(fh.evalf(subs={x: x_n}))  # 第k次迭代误差的大小
        print('%3d %22.15f %22.15f' % (k + 1, x_n, errval))  # 分别输出迭代次数，方程的近似根以及迭代误差
        if errval <= eps:
            break
    if k + 1 <= maxIter - 1:
        print('方程在满足精度' + str(eps) + '的条件下，近似解为：'
              + str(x_n) + ',误差为：' + str(errval))
    else:
        print('不动点迭代法求解方程的根，已经达到最大迭代次数，也可能不收敛或精度过高...')
    return None


if __name__ == '__main__':
    fh = fun()
    plot(fh)
    x0 = float(input('请输入迭代初值：'))  # input函数总是以字符串的形式返回
    eps = float(input('请输入误差精度要求：'))  # 方程解的精度要求是近似解与真值之间的误差
    maxIter = int(input('请输入最大迭代次数：'))  # 方程一般会迭代无数次，必须定义其迭代的次数，以求收敛
    fixedPoint(x0, eps, maxIter)
    print('方程为：', '%30s' % (str(fun())))
    print('迭代函数为：', '%20s' % ('20 / (x**2 + 2 * x + 10)'))
