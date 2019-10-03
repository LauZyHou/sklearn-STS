from sklearn import linear_model

if __name__ == '__main__':
    """ridge regression"""
    reg = linear_model.Ridge(alpha=0.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print(reg.coef_)
    """ Generalized Cross-Validation"""
    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print(reg.coef_, reg.alpha_)
