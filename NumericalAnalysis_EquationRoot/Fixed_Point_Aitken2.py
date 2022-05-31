from sympy import *


def Aitken(x0, eps, iterNum, Phi):
    xk_1 = x0
    errval = 0
    print('%3s %10s %18s' % ('迭代次数', '方程近似值', '迭代误差'))
    for k in range(iterNum):
        y = Phi(xk_1)
        z = Phi(y)
        if (z - 2 * y + xk_1) != 0:
            xk = xk_1 - (y - xk_1) ** 2 / (z - 2 * y + xk_1)
            errval = abs(xk - xk_1)
            print('%3d %22.15f %22.15f' % (k + 1, xk, errval))
            if errval < eps:
                return xk
            else:
                xk_1 = xk
        else:
            return xk
    print('方法失败')
    return 0


def Phi(x):
    return 2 - (pow(x, 3) + 2 * pow(x, 2))/10


if __name__ == '__main__':
    x = symbols('x')
    x0 = float(input('请输入迭代初值：'))  # input函数总是以字符串形式返回
    eps = float(input('请输入迭代误差精度要求：'))  # 方程误差精度要求为迭代近似值与真值之间的差值
    iterNum = int(input('请输入最大迭代次数：'))  # 最大迭代次数限制了方程的收敛
    Phi(x)
    Aitken(x0, eps, iterNum, Phi)
