from sklearn import linear_model

if __name__ == '__main__':
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print(reg.coef_)
