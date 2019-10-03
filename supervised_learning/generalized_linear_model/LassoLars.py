# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 14:07
# @Author  : LauZyHou
# @File    : LARS Lasso
from sklearn import linear_model

if __name__ == '__main__':
    reg = linear_model.LassoLars(alpha=.1)
    reg.fit([[0, 0], [1, 1]], [0, 1])
    print(reg.coef_)
