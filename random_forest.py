import pandas as pd
import data
import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns


def random_forest(x_train, y_train, x_test, y_test):
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    rf = RandomForestRegressor()

    estimators_rf = np.linspace(1, 20, num=5, dtype=int)

    r2 = []
    rmse = []
    mae = []

    print()
    for n in estimators_rf:
        rf.set_params(n_estimators=n)
        rf.fit(x_train, y_train)
        prediction_rf = rf.predict(x_test)

        pred['RF_Prediction'] = prediction_rf

        print(n, prediction_rf)
        print("RANDOM FOREST WITH " + str(n) + " TREES")
        print("R²: " + str(r2_score(y_test, prediction_rf)))
        r2.append(r2_score(y_test, prediction_rf))
        print("RMSE: " + str(mean_squared_error(y_test, prediction_rf, squared=False)))
        rmse.append(mean_squared_error(y_test, prediction_rf, squared=False))
        print("MAE: " + str(mean_absolute_error(y_test, prediction_rf)))
        mae.append(mean_absolute_error(y_test, prediction_rf))
        print()

        ax1 = pred.plot(x='date', y='RefSt')
        pred.plot(x='date', y='RF_Prediction', ax=ax1, title='Random Forest for ' + str(n) + ' trees.')
        plt.show()

        sns_rf = sns.lmplot(x='RefSt', y='RF_Prediction', data=pred, fit_reg=True,
                            line_kws={'color': 'orange'}).set(title='Random Forest for ' + str(n) + ' trees.')
        sns_rf.set(ylim=(-2, 3))
        sns_rf.set(xlim=(-2, 3))
        plt.show()

    utils.table_creation(['Number of trees', 'R^2', 'RMSE', 'MAE'], [estimators_rf, r2, rmse, mae],
                         'random_forest_table.txt')

    plt.title("Random Forest. Metrics vs  number of estimators (trees)")
    plt.xlabel('Number of trees')
    plt.ylabel('Metric value')
    plt.plot(estimators_rf, r2, color='red', label="R^2")
    plt.plot(estimators_rf, rmse, color='blue', label="RMSE")
    plt.plot(estimators_rf, mae, color='green', label="MAE")
    plt.legend(loc="center left")
    plt.show()
