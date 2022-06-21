import pandas as pd
import numpy as np
import data
import utils
import math
import seaborn as sns  # for scatter plot
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def kernel_rbf(x, y):
    # divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    alphas = np.arange(0.1, 1.1, 0.1)
    r2 = []
    mse = []
    mae = []
    for a in alphas:
        clf = KernelRidge(kernel='rbf', alpha=a)
        clf.fit(x_train, y_train)
        r2_scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='r2')
        mae_score = cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_absolute_error')
        mse_score = cross_val_score(clf, x_train, y_train, cv=10)
        r2.append(r2_scores.mean())
        mse.append(mse_score.mean())
        mae.append(mae_score.mean())

    rmse = [math.sqrt(1 - x) for x in mse]
    mae = [-1 * x for x in mae]
    print(r2)
    print(rmse)
    utils.table_creation(['Alpha', 'R^2', 'RMSE', 'MAE'], [alphas, r2, rmse, mae],
                         'rbf_table.txt')

    # plot errors
    plt.title("R-squared")
    plt.xlabel('Lambda')
    plt.ylabel('R^2')
    plt.plot(alphas, r2, color='red')
    plt.savefig("img/rbf_r2")
    plt.clf()

    plt.title("Root Mean Squared Error")
    plt.xlabel('Lambda')
    plt.ylabel('RMSE')
    plt.plot(alphas, rmse, color='blue')
    plt.savefig("img/rbf_rmse")
    plt.clf()

    plt.title("Mean Absoulte Error")
    plt.xlabel('Lambda')
    plt.ylabel('MAE')
    plt.plot(alphas, mae, color='black')
    plt.savefig("img/rbf_mae")
    plt.clf()

    # best alpha is the one with higher R^2
    best_n = alphas[r2.index(max(r2))]
    print(best_n)

    model = KernelRidge(alpha=best_n)
    model.fit(x_train, y_train)
    mpred = model.predict(x_test)
    acc = model.score(x_test, y_test)
    pred['Pred'] = mpred

    print("RBF PREDICTION")
    print("R^2: ", str(r2_score(y_test, mpred)))
    print("RMSE: ", str(mean_squared_error(y_test, mpred, squared=False)))
    print("MAE: ", str(mean_absolute_error(y_test, mpred)))
    print("Accuracy: ", str(acc * 100))

    # Plots
    ax1 = pred.plot(x='date', y='RefSt', color='red')
    pred.plot(x='date', y='Pred', ax=ax1, title='RBF for alpha ' + str(best_n), color='blue')
    plt.savefig("img/RBF_pred")
    plt.clf()

    sns_rf = sns.lmplot(x='RefSt', y='Pred', data=pred, fit_reg=True, height=5, aspect=1.5,
                        line_kws={'color': 'orange'})
    sns_rf.fig.suptitle('RBF for alpha' + str(best_n))
    sns_rf.set(ylim=(-2, 3))
    sns_rf.set(xlim=(-2, 3))
    plt.savefig("img/RBF_line")
    plt.clf()



