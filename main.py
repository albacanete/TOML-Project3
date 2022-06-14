import data
import pandas as pd
from datetime import datetime
import plots

if __name__ == "__main__":
    # CLEAN data before plotting (I.E. dates to datetime, big numbers to numeric)
    data.new_PR_data_inner['date'] = pd.to_datetime(data.new_PR_data_inner['date'], format='%Y-%m-%d %H:%M:%S')
    data.new_PR_data_inner['date'] = data.new_PR_data_inner['date'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    data.new_PR_data_inner['Sensor_O3'] = data.new_PR_data_inner['Sensor_O3'].str.replace(".", "", regex=True).astype(float)
    data.new_PR_data_inner['Sensor_O3'] = pd.to_numeric(data.new_PR_data_inner['Sensor_O3'])

    plots.plot_O3(data.new_PR_data_inner)
    plots.scatter_ref(data.new_PR_data_inner)
    plots.plot_metrics(data.new_PR_data_inner)