import matplotlib.pyplot as plt


def plot_O3(data):
    # O3 sensor data against date
    data.plot(x='date', y='Sensor_O3')
    plt.show()

    # O3 ref station against date
    data.plot(x='date', y='RefSt', color='red')
    plt.gcf().autofmt_xdate()
    plt.show()


def scatter_ref(data):
    # O3 sensor against ref station
    data.plot.scatter(x='Sensor_O3', y='RefSt', color='green')
    plt.show()

    # Normalize the data
    data['Sensor_O3'] = (data['Sensor_O3'] - data['Sensor_O3'].mean()) / \
                                     data['Sensor_O3'].std()
    data['RefSt'] = (data['RefSt'] - data['RefSt'].mean()) / data[
        'RefSt'].std()

    # O3 sensor agains ref station normalized
    normalized_plt = data.plot.scatter(x='Sensor_O3', y='RefSt', color='green')
    normalized_plt.set_xlabel("Sensor_O3 normalized")
    normalized_plt.set_ylabel("RefSt normalized")
    plt.show()


def plot_metrics(data):
    # select only needed metrics
    columns_plot = data.columns[3:]  # Select only necessary columns

    # O3 sensor against metrics
    for i in columns_plot:
        data.plot.scatter(x='Sensor_O3', y=i)
        plt.show()

    # O3 ref station against metrics
    for i in columns_plot:
        data.plot.scatter(x='RefSt', y=i, color='red')
        plt.show()
