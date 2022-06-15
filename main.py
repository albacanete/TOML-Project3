import data
import pandas as pd
import data_observation
import plots
import MLR

if __name__ == "__main__":
    # remove dots from thousands in O3 sensors data
    data.new_PR_data_inner['Sensor_O3'] = data.new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    data.new_PR_data_inner['Sensor_O3'] = pd.to_numeric(data.new_PR_data_inner['Sensor_O3'])

    plots.plot_O3(data.new_PR_data_inner)
    plots.scatter_ref(data.new_PR_data_inner)
    plots.plot_metrics(data.new_PR_data_inner)

    means = data_observation.get_means()
    print(means)
    cov = data_observation.get_cov_matrix()
    print(cov)