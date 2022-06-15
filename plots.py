import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)


def plot_O3(data):
    # O3 sensor data against date
    data.plot(x='date', y='Sensor_O3')
    plt.title("O3 sensor data (KOhms) vs date")
    plt.ylabel("KOhms")
    plt.savefig("img/03sensor_date2")
    plt.clf()
    # plt.show()

    # O3 ref station against date
    data.plot(x='date', y='RefSt', color='red')
    # plt.gcf().autofmt_xdate()
    plt.title("O3 ref station data (µgr/m^3) vs date")
    plt.ylabel("µgr/m^3")
    plt.savefig("img/O3ref_date2")
    plt.clf()
    # plt.show()


def scatter_ref(data):
    # O3 sensor against ref station
    data.plot.scatter(x='Sensor_O3', y='RefSt', color='blue')
    plt.title("O3 sensor data (KOhms) vs O3 ref station data (µgr/m^3)")
    plt.savefig("img/O3sensor_03ref")
    plt.clf()
    # plt.show()

    # Normalize the data
    data['Sensor_O3_norm'] = (data['Sensor_O3'] - data['Sensor_O3'].mean()) / \
                                     data['Sensor_O3'].std()
    data['RefSt_norm'] = (data['RefSt'] - data['RefSt'].mean()) / data[
        'RefSt'].std()

    # O3 sensor agains ref station normalized
    normalized_plt = data.plot.scatter(x='Sensor_O3_norm', y='RefSt_norm', color='blue')
    normalized_plt.set_xlabel("Sensor_O3 normalized")
    normalized_plt.set_ylabel("RefSt normalized")
    plt.axline([0, 0], [1, 1], color='red', linestyle="--")
    plt.title("Normalization of O3 sensor data (KOhms) vs O3 ref station data (µgr/m^3)")
    plt.savefig("img/O3sensor_03ref_norm")
    plt.clf()
    # plt.show()


def plot_metrics(data):
    # select only needed metrics
    columns_plot = data.columns[3:]  # Select only necessary columns

    # O3 sensor against metrics: temperature, relative humidity, NO2, NO, SO2
    for i in columns_plot:
        data.plot.scatter(x='Sensor_O3', y=i)
        plt.title("Normalization of the O3 sensor vs " + i)
        name = "03sensor_" + i
        plt.savefig("img/" + name)
        # plt.show()

    # O3 ref station against metrics
    for i in columns_plot:
        data.plot.scatter(x='RefSt', y=i, color='green')
        plt.title("Normalization of the O3 ref station vs " + i)
        name = "O3ref_" + i
        plt.savefig("img/" + name)
        # plt.show()
