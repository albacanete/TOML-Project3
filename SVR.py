import pandas as pd
import numpy as np
import seaborn as sns  # for scatter plot
import matplotlib.pyplot as plt
import utils
import data

from sklearn.svm import SVR

# C = np.linspace(0, 1000)    # regularization param
# gamma = np.linspace(1, 10)
# epsilon = np.linspace(0, 1)
# kernel = ["rbf", "poly", "linear"]
# degree = [1, 2, 3, 4, 5]


def svr(x_train, y_train, x_test, y_test):
    # TODO: perform grid search to find best hyperparameters. This is just a test
    C = 100
    degree = 2
    gamma = "scale"
    kernel = "rbf"

    # create model
    svr_model = SVR(kernel=kernel, C=C, degree=degree, gamma=gamma)

    # fit model
    svr_model.fit(x_train, y_train)

    # predict model
    svr_pred = svr_model.predict(x_test)

    # Plot linear
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['SVR_Pred'] = svr_pred
    pred['date'] = data.new_PR_data_inner['date']
    ax = pred.plot(x='date', y='RefSt')
    pred.plot(x='date', y='SVR_Pred', ax=ax, title='Support Vector Regression')
    plt.show()

    # Plot regression
    sns.lmplot(x='RefSt', y='SVR_Pred', data=pred, fit_reg=True, line_kws={'color': 'orange'})
    plt.show()

    # get and print loss functions
    r2, rmse, mae = utils.loss_functions(pred["RefSt"], pred["SVR_Pred"])