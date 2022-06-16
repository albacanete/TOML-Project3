import data
import pandas as pd
import plots
import seaborn as sns # for scatter plot
import matplotlib.pyplot as plt
import utils
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


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

    # FORWARD SUBSET SELECTION
    print("******* FORWARD SUBSET SELECTION *******")
    x = norm_data.drop(['date', 'RefSt'], axis=1)
    y = norm_data['RefSt']
    x_train, x_test, y_train, y_test = utils.train_test_split(x, y)
    k = 3
    best_features = utils.forward_subset_selection(x_train, y_train, k)
    print(best_features)  # ['Sensor_O3', 'Temp', 'RelHum']

    best_x_train = x_train[['Sensor_O3', 'Temp', 'RelHum']]
    best_x_test = x_test[['Sensor_O3', 'Temp', 'RelHum']]
    """
    # Calculate multiple linear regression model
    linear_model = LinearRegression()
    linear_model.fit(best_x_train, y_train)
    # print coefficients
    print("Intercept: ",  linear_model.intercept_)
    print("Coefficients: ",  linear_model.coef_)
    # predictions
    norm_data["MRL"] = linear_model.intercept_ \
                + linear_model.coef_[0] * norm_data["Sensor_O3"] \
                + linear_model.coef_[1] * norm_data["Temp"] \
                + linear_model.coef_[2] * norm_data["RelHum"]
    # plot
    norm_data[["RefSt", "MRL"]].plot()
    plt.show()"""

    # RIDGE REGRESSION
    print("******* RIDGE REGRESSION *******")
    rm = linear_model.Ridge()

    alphas = [1, 5, 10, 50, 100, 250, 500]
    coef = []
    r2 = []
    rmse = []
    mae = []
    for a in alphas:
        rm.set_params(alpha=a)
        rm.fit(best_x_train, y_train)

        print("Alpha: ", a)
        coef.append(rm.coef_)
        print("Coefficients: ", rm.coef_)
        pred = rm.predict(best_x_test)
        print("R^2:", r2_score(y_test, pred))
        r2.append(r2_score(y_test, pred))
        print("RMSE: ", mean_squared_error(y_test, pred, squared=False))
        rmse.append(mean_squared_error(y_test, pred, squared=False))
        print("MAE: ", mean_absolute_error(y_test, pred))
        mae.append(mean_absolute_error(y_test, pred))

    # plot coefficients with alphas
    ax = plt.gca()
    ax.plot(alphas, coef)
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title("Plot of coefficients for different alphas")
    ax.legend(("Sensor O3 coef", "Temp coef", "RelHum coef"))
    plt.savefig("img/ridge_coefs")
    plt.clf()
    # plt.show()

    # plot errors
    plt.title("R^2, Root Mean Square Error and Mean Absolute Error")
    plt.xlabel('Different alphas')
    plt.ylabel('Error')
    plt.plot(alphas, r2, color='red')
    plt.plot(alphas, rmse, color='black')
    plt.plot(alphas, mae, color='green')
    plt.legend(("R^2", "RMSE", "MAE"))
    plt.show()

    # choose alpha
    rm.set_params(alpha=10)
    rm.fit(best_x_train, y_train)

    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Ridge_Pred'] = rm.intercept_ \
        + rm.coef_[0] * best_x_test['Sensor_O3'] \
        + rm.coef_[1] * best_x_test['Temp'] \
        + rm.coef_[2] * best_x_test['RelHum']
    pred['date'] = data.new_PR_data_inner['date']
    # Plot estimated O3 against date O3 reference data
    plt.title("MLR against RefSt")
    plt.xlabel("date")
    pred[['RefSt']].plot()
    plt.xticks(rotation=20)

    # sns.lmplot(x='RefSt', y='Ridge_Pred', data=pred, fit_reg=True, line_kws={'color': 'orange'})
    plt.show()
