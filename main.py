import data
import pandas as pd
import plots
import utils


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

    # forward subset selection
    x = norm_data.drop(['date', 'RefSt'], axis=1)
    y = norm_data['RefSt']
    x_train, x_test, y_train, y_test = utils.train_test_split(x, y)
    k = 3
    best_features = utils.forward_subset_selection(x_train, y_train, k)
    print(best_features)
