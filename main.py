import data
import pandas as pd
import plots
import utils
from sklearn.model_selection import train_test_split

import MLR
import SVR
import RBF
import KNN
import KNN
import RF


if __name__ == "__main__":
    # remove dots from thousands in O3 sensors data
    data.new_PR_data_inner['Sensor_O3'] = data.new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    data.new_PR_data_inner['Sensor_O3'] = pd.to_numeric(data.new_PR_data_inner['Sensor_O3'])

    # plots.plot_O3(data.new_PR_data_inner)
    # plots.scatter_ref(data.new_PR_data_inner)
    # plots.plot_metrics(data.new_PR_data_inner)

    # create dataframe with normalized data
    norm_data = data.new_PR_data_inner.copy(deep=True)
    norm_data['RefSt'] = utils.normalize(norm_data['RefSt'])
    norm_data['Sensor_O3'] = utils.normalize(norm_data['Sensor_O3'])
    norm_data['Temp'] = utils.normalize(norm_data['Temp'])
    norm_data['RelHum'] = utils.normalize(norm_data['RelHum'])
    norm_data['Sensor_NO2'] = utils.normalize(norm_data['Sensor_NO2'])
    norm_data['Sensor_NO'] = utils.normalize(norm_data['Sensor_NO'])
    norm_data['Sensor_SO2'] = utils.normalize(norm_data['Sensor_SO2'])
    print(norm_data)

    # means = utils.get_means(norm_data)
    # print(means)
    # cov = utils.get_cov_matrix(norm_data)
    # print(cov)

    # division of the dataset
    x = norm_data.drop(['date', 'RefSt'], axis=1)
    y = norm_data['RefSt']

    # Multiple linear regression
    MLR.forward_subset(x, y)
    MLR.ridge_regression(x, y)
    MLR.lasso(x, y)

    # KNN
    # KNN.k_neighbors(best_x_train, y_train, best_x_test, y_test, best_x_val, y_val)

    # Kernels
    # RBF.polynomial_kernel(x_train, y_train, x_test, y_test)
    # RBF.gaussian_kernel(x_train, y_train, x_test, y_test)

    # Random forest
    # RF.random_forest(x_train, y_train, x_test, y_test)

    # Support vector regression
    # SVR.svr(best_x_train, y_train, best_x_test, y_test)
