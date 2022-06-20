import pandas as pd
import seaborn as sns  # for scatter plot
import matplotlib.pyplot as plt
import data
import utils

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def k_neighbors(x_train, y_train, x_test, y_test):
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15]

    r2 = []
    rmse = []
    mae = []

    for n in n_neighbors:
        model = KNeighborsRegressor(n_neighbors=n)
        model.fit(x_train, y_train)
        mpred = model.predict(x_test)

        pred['KNN_Prediction'] = mpred

        print(n, mpred)
        print("KNN " + str(n) + " neighbors")
        print("R²: " + str(r2_score(y_test, mpred)))
        r2.append(r2_score(y_test, mpred))
        print("RMSE: " + str(mean_squared_error(y_test, mpred, squared=False)))
        rmse.append(mean_squared_error(y_test, mpred, squared=False))
        print("MAE: " + str(mean_absolute_error(y_test, mpred)))
        mae.append(mean_absolute_error(y_test, mpred))

        ax1 = pred.plot(x='date', y='RefSt')
        pred.plot(x='date', y='KNN_Prediction', ax=ax1, title='KNN for ' + str(n) + ' neighbors.')
        label = "KNN_" + str(n)
        plt.savefig("img/" + label)
        plt.clf()

        sns_rf = sns.lmplot(x='RefSt', y='KNN_Prediction', data=pred, fit_reg=True,
                            line_kws={'color': 'orange'}).set(title='KNN for ' + str(n) + ' neighbors.')
        sns_rf.set(ylim=(-2, 3))
        sns_rf.set(xlim=(-2, 3))
        label = "KNN_line_" + str(n)
        plt.savefig("img/" + label)
        plt.clf()

    utils.table_creation(['Number of neighbours', 'R^2', 'RMSE', 'MAE'], [n_neighbors, r2, rmse, mae],
                         'knn_table.txt')

    plt.title("KNN erros vs  number of neighbors")
    plt.xlabel('Number of neighbors')
    plt.ylabel('Error value')
    plt.plot(n_neighbors, r2, color='red', label="R^2")
    plt.plot(n_neighbors, rmse, color='blue', label="RMSE")
    plt.plot(n_neighbors, mae, color='green', label="MAE")
    plt.legend(loc="center left")
    plt.savefig("img/KNN_errors")
    plt.clf()
