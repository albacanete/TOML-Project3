import pandas as pd
import numpy as np
import seaborn as sns  # for scatter plot
import matplotlib.pyplot as plt
import utils
import data

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def svr(x_train, y_train, x_test, y_test):
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    # TODO: perform grid search to find best hyperparameters. This is just a test
    # C = 100
    # degree = 2
    # gamma = "scale"
    # kernel = "rbf"

    # Performing hyper-parameters grid search
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    degree = [1, 2, 3, 4, 5, 6, 7]
    gamma = ["auto", "scale"]
    kernel = ["rbf", "poly", "linear"]

    param_grid = {
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,
        'C': C,
    }

    scoring_cols = [
        'param_kernel',
        'param_C',
        'param_degree',
        'param_gamma',
        'mean_test_mae',
        'mean_test_mse',
        'mean_test_r2',
    ]

    scoring_dict = {
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error',
        'r2': 'r2',
    }

    cvModel = GridSearchCV(
        estimator=SVR(),
        scoring=scoring_dict,
        param_grid=param_grid,
        refit='mae',
        cv=10,
        n_jobs=-1,
        return_train_score=False
    )

    cvModel = cvModel.fit(x_train, y_train)

    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    scores = pd.DataFrame(cvModel.cv_results_).sort_values(by='mean_test_mse', ascending=False)[scoring_cols].head()
    print(scores)

    """"
    BEST SCORES ARE:
            param_kernel param_C  param_degree param_gamma  mean_test_mae  mean_test_mse  mean_test_r2
    168          rbf      10           1        auto         -0.186          -0.063         0.935  
    """
    k = "rbf"
    c = 10
    d = 1
    g = "auto"
    # create model
    svr_model = SVR(kernel=k, C=c, degree=d, gamma=g)
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

