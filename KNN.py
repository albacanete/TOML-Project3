import pandas as pd
import seaborn as sns  # for scatter plot
import matplotlib.pyplot as plt
import data

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


def k_neighbors(x_train, y_train, x_test, y_test):
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15]

    param_grid = {
        'n_neighbors': n_neighbors,
    }

    scoring_cols = [
        'param_n_neighbors',
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
        estimator=KNeighborsRegressor(),
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
    scores = pd.DataFrame(cvModel.cv_results_).sort_values(by='mean_test_mae', ascending=False)[scoring_cols].head()
    print(scores)

    # Plot linear
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['KNN_Pred'] = cvModel.predict(x_test)
    pred['date'] = data.new_PR_data_inner['date']
    ax = pred.plot(x='date', y='RefSt')
    pred.plot(x='date', y='KNN_Pred', ax=ax, title='K-Nearest Neighbor')
    plt.show()

    # Plot regression
    sns.lmplot(x='RefSt', y='KNN_Pred', data=pred, fit_reg=True, line_kws={'color': 'orange'})
    plt.show()

