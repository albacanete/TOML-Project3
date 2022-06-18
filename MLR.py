import pandas as pd
import seaborn as sns  # for scatter plot
import matplotlib.pyplot as plt
import utils
import data

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import linear_model


def ridge_regression(x_train, y_train, x_test, y_test):
    print("******* RIDGE REGRESSION *******")
    rm = linear_model.Ridge()

    alphas = [1, 5, 10, 50, 100, 250, 500]
    coef = []
    r2 = []
    rmse = []
    mae = []
    for a in alphas:
        rm.set_params(alpha=a)
        rm.fit(x_train, y_train)

        print("Alpha: ", a)
        coef.append(rm.coef_)
        print("Coefficients: ", rm.coef_)
        pred = rm.predict(x_test)
        r2.append(r2_score(y_test, pred))
        rmse.append(mean_squared_error(y_test, pred, squared=False))
        mae.append(mean_absolute_error(y_test, pred))

    utils.table_creation(['Alpha', 'R^2', 'RMSE', 'MAE'], [alphas, r2, rmse, mae], 'ridge_regression_errors.txt')
    utils.table_creation(['Alpha', 'betas'], [alphas, coef], 'ridge_regression_coefs.txt')

    # plot coefficients with alphas
    ax = plt.gca()
    ax.plot(alphas, coef)
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title("Plot of coefficients for different alphas")
    ax.legend(("Sensor O3 coef", "Temp coef", "RelHum coef"))
    plt.savefig("img/ridge_coefs")
    plt.clf()
    # plt.show()

    # plot errors
    plt.title("R^2, Root Mean Square Error and Mean Absolute Error")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, r2, color='red')
    plt.plot(alphas, rmse, color='black')
    plt.plot(alphas, mae, color='green')
    plt.legend(("R^2", "RMSE", "MAE"))
    plt.show()

    # choose alpha
    rm.set_params(alpha=10)
    rm.fit(x_train, y_train)

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Ridge_Pred'] = rm.predict(x_test)
    pred['date'] = data.new_PR_data_inner['date']

    sns.lmplot(x='RefSt', y='Ridge_Pred', data=pred, fit_reg=True, line_kws={'color': 'orange'})
    plt.show()


def lasso(x_train, y_train, x_test, y_test):
    print("******* LASSO *******")
