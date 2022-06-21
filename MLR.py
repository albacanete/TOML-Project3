import pandas as pd
import numpy as np
import seaborn as sns  # for scatter plot
import matplotlib.pyplot as plt
import utils
import data

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 6)


def forward_subset(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    # number of features
    k = 3

    # Check if missing values for the data
    lreg = LinearRegression()

    # FORWARD by R2
    sfs_r2 = sfs(lreg, k_features=k, forward=True, scoring='r2').fit(x_train, y_train)

    best_features = list(sfs_r2.k_feature_names_)
    print(best_features)

    best_x_train = x_train[best_features]
    best_x_test = x_test[best_features]

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    model = LinearRegression()
    model.fit(best_x_train, y_train)
    mpred = model.predict(best_x_test)
    pred["MLR_Pred"] = mpred

    # compute errors
    print("BEST SUBSET PREDICTION")
    r2 = r2_score(y_test, mpred)
    print("R^2: ", str(r2))
    rmse = mean_squared_error(y_test, mpred, squared=False)
    print("RMSE: ", str(rmse))
    mae = mean_absolute_error(y_test, mpred)
    print("MAE: ", str(mae))

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='MLR_Pred', ax=ax1, title='Forward subset selection for ' + str(best_features) + ' features.',
              color='blue')
    # plt.show()
    plt.savefig("img/MLR_forward")
    plt.clf()

    sns_rf = sns.lmplot(x='RefSt', y='MLR_Pred', data=pred, fit_reg=True,
                        line_kws={'color': 'orange'})
    sns_rf.fig.suptitle('Forward subset selection ' + str(best_features) + ' features.')
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    plt.show()
    # plt.savefig("img/MLR_forward_line")
    # plt.clf()


def ridge_regression(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

    alphas = [0, 1, 5, 10, 50, 100, 250, 500]
    coef = []
    r2 = []
    rmse = []
    mae = []
    for a in alphas:
        rm = linear_model.Ridge()
        rm.set_params(alpha=a)
        rm.fit(x_val, y_val)

        print("Alpha: ", a)
        coef.append(rm.coef_)
        print("Intercept: ", rm.intercept_)
        print("Coefficients: ", rm.coef_)
        pred = rm.predict(x_test)
        r2.append(r2_score(y_test, pred))
        rmse.append(mean_squared_error(y_test, pred, squared=False))
        mae.append(mean_absolute_error(y_test, pred))

    utils.table_creation(['Alpha', 'R^2', 'RMSE', 'MAE'], [alphas, r2, rmse, mae], 'ridge_regression_errors.txt')
    utils.table_creation(['Alpha', 'coefs'], [alphas, coef], 'ridge_regression_coefs.txt')

    # plot errors
    plt.title("R^2, Root Mean Square Error and Mean Absolute Error")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, r2, color='red')
    plt.plot(alphas, rmse, color='black')
    plt.plot(alphas, mae, color='green')
    plt.legend(("R^2", "RMSE", "MAE"))
    plt.savefig("img/ridge_errors")
    plt.clf()
    # plt.show()

    # plot coefficients with alphas
    ax = plt.gca()
    ax.plot(alphas, coef)
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title("Plot of coefficients for different alphas")
    ax.legend(("Sensor O3 coef", "Temp coef", "RelHum coef", "Sensor_NO2", "Sensor_NO", "Sensor_SO2"))
    plt.savefig("img/ridge_coefs")
    plt.clf()
    # plt.show()

    # best alpha is the one with less R^2
    min_r2 = max(r2)
    best_a = alphas[r2.index(min_r2)]

    rm = linear_model.Ridge()
    rm.set_params(alpha=1)
    rm.fit(x_train, y_train)
    rmpred = rm.predict(x_test)
    acc = rm.score(x_test, y_test)

    print("RIDGE PREDICTION")
    print("R^2: ", str(r2_score(y_test, rmpred)))
    print("RMSE: ", str(mean_squared_error(y_test, rmpred, squared=False)))
    print("MAE : ", str(mean_absolute_error(y_test, rmpred)))


    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Ridge_Pred'] = rmpred
    pred['date'] = data.new_PR_data_inner['date']

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Ridge_Pred', ax=ax1, title='Ridge regression for alpha 1', color='blue')
    label = "Ridge_1"
    plt.savefig("img/" + label)
    plt.clf()

    sns_r = sns.lmplot(x='RefSt', y='Ridge_Pred', data=pred, fit_reg=True,  height=5, aspect=1.5,
                       line_kws={'color': 'orange'})
    sns_r.fig.suptitle('Ridge regression for alpha 1')
    sns_r.set(ylim=(-2, 3))
    sns_r.set(xlim=(-2, 3))
    # plt.show()
    label = "Ridge_line_1"
    plt.savefig("img/" + label)
    plt.clf()


def lasso(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

    alphas = np.arange(0.1, 1, 0.1)
    coef = []
    r2 = []
    rmse = []
    mae = []
    for a in alphas:
        rm = linear_model.Lasso()
        rm.set_params(alpha=a)
        rm.fit(x_val, y_val)

        print("Alpha: ", a)
        coef.append(rm.coef_)
        print("Intercept: ", rm.intercept_)
        print("Coefficients: ", rm.coef_)
        pred = rm.predict(x_test)
        r2.append(r2_score(y_test, pred))
        rmse.append(mean_squared_error(y_test, pred, squared=False))
        mae.append(mean_absolute_error(y_test, pred))

    utils.table_creation(['Alpha', 'R^2', 'RMSE', 'MAE'], [alphas, r2, rmse, mae], 'lasso_errors.txt')
    utils.table_creation(['Alpha', 'coefs'], [alphas, coef], 'lasso_coefs.txt')

    # plot errors
    plt.title("R^2, Root Mean Square Error and Mean Absolute Error")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, r2, color='red')
    plt.plot(alphas, rmse, color='black')
    plt.plot(alphas, mae, color='green')
    plt.legend(("R^2", "RMSE", "MAE"))
    plt.savefig("img/lasso_errors")
    plt.clf()
    # plt.show()

    # plot coefficients with alphas
    ax = plt.gca()
    ax.plot(alphas, coef)
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title("Plot of coefficients for different alphas")
    ax.legend(("Sensor O3 coef", "Temp coef", "RelHum coef", "Sensor_NO2", "Sensor_NO", "Sensor_SO2"))
    plt.savefig("img/lasso_coefs")
    plt.clf()
    # plt.show()

    # best alpha is the one with less R^2
    min_r2 = max(r2)
    best_a = alphas[r2.index(min_r2)]

    rm = linear_model.Lasso()
    rm.set_params(alpha=best_a)
    rm.fit(x_train, y_train)
    rmpred = rm.predict(x_test)

    print("LASSO PREDICTION")
    print("R^2: ", str(r2_score(y_test, rmpred)))
    print("RMSE: ", str(mean_squared_error(y_test, rmpred, squared=False)))
    print("MAE : ", str(mean_absolute_error(y_test, rmpred)))

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Pred'] = rmpred
    pred['date'] = data.new_PR_data_inner['date']

    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Pred', ax=ax1, title='LASSO for alpha ' + str(best_a), color='blue')
    plt.savefig("img/LASSO_pred")
    plt.clf()

    sns_r = sns.lmplot(x='RefSt', y='Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                       line_kws={'color': 'orange'})
    sns_r.fig.suptitle('LASSO for alpha ' + str(best_a))
    sns_r.set(ylim=(-2, 3))
    sns_r.set(xlim=(-2, 3))
    # plt.show()
    plt.savefig("img/LASSO_line")
    plt.clf()
