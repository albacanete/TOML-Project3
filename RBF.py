import pandas as pd
import data
import utils
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def polynomial_kernel(x_train, y_train, x_test, y_test):
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    r2 = []
    rmse = []
    mae = []
    degree = [i for i in range(1, 10)]
    for d in degree:
        kernel_poly = KernelRidge(kernel='poly', degree=d).fit(x_train, y_train)
        pred_kernel_poly_model = kernel_poly.predict(x_test)
        print("R^2:", r2_score(y_test, pred_kernel_poly_model))
        r2.append(r2_score(y_test, pred_kernel_poly_model))
        print("RMSE: ", mean_squared_error(y_test, pred_kernel_poly_model, squared=False))
        rmse.append(mean_squared_error(y_test, pred_kernel_poly_model, squared=False))
        print("MAE: ", mean_absolute_error(y_test, pred_kernel_poly_model))
        mae.append(mean_absolute_error(y_test, pred_kernel_poly_model))
        if d == 7:  # Best degree for the kernel function
            pred['Best_Kernel_Poly_Pred'] = pred_kernel_poly_model
        else:
            pred['Kernel_Poly_Pred'] = pred_kernel_poly_model

        # Plots
        ax = pred.plot(x='date', y='RefSt')
        pred.plot(x='date', y='Kernel_Poly_Pred', ax=ax,
                       title='Polynomial Kernel Ridge Regression with degree=' + str(d))
        plt.show()
        # plt.savefig(path_kernel_ridge_regression_plots + str("models/kernel_poly_model_d-" + str(d) + ".png"),
        # bbox_inches='tight') plt.clf()

    kr_stats = pd.DataFrame({'degree': degree, 'R^2': r2, 'RMSE': rmse, 'MAE': mae})
    kr_stats = kr_stats.set_index('degree')  # index column (X axis for the plots)

    kr_stats[["R^2"]].plot()
    plt.show()
    # plt.savefig(path_kernel_ridge_regression_plots + str("error_metrics/r_squared_polynomial.png"),
    # bbox_inches='tight') plt.clf()

    kr_stats[["RMSE"]].plot()
    plt.show()
    # plt.savefig(path_kernel_ridge_regression_plots + str("error_metrics/rmse_polynomial.png"), bbox_inches='tight')
    # plt.clf()

    kr_stats[["MAE"]].plot()
    plt.show()
    # plt.savefig(path_kernel_ridge_regression_plots + str("error_metrics/mae_polynomial.png"), bbox_inches='tight')
    # plt.clf()

    # Create the table and save it to a file
    utils.table_creation(['Degree value', 'R²', 'RMSE', 'MAE'], [degree, r2, rmse, mae], 'kernel_ridge_regression_poly.txt')


def gaussian_kernel(x_train, y_train, x_test, y_test):
    pred = pd.DataFrame()
    pred['RefSt'] = y_test
    pred['Sensor_O3'] = x_test['Sensor_O3']
    pred['date'] = data.new_PR_data_inner['date']

    kernel_rbf = KernelRidge(kernel="rbf").fit(x_train, y_train)
    pred_kernel_gauss_model = kernel_rbf.predict(x_test)

    r2 = []
    rmse = []
    mae = []
    print("R²:", r2_score(y_test, pred_kernel_gauss_model))
    r2.append(r2_score(y_test, pred_kernel_gauss_model))
    print("RMSE: ", mean_squared_error(y_test, pred_kernel_gauss_model, squared=False))
    rmse.append(mean_squared_error(y_test, pred_kernel_gauss_model, squared=False))
    print("MAE: ", mean_absolute_error(y_test, pred_kernel_gauss_model))
    mae.append(mean_absolute_error(y_test, pred_kernel_gauss_model))
    pred['Kernel_Gauss_Pred'] = pred_kernel_gauss_model

    # Plot
    ax = pred.plot(x='date', y='RefSt')
    pred.plot(x='date', y='Kernel_Gauss_Pred', ax=ax,
                   title='Gaussian Kernel Ridge Regression')
    plt.show()
    # plt.savefig(path_kernel_ridge_regression_plots + str("models/kernel_gauss_model.png"), bbox_inches='tight')
    # plt.clf()

    # Create the table and save it to a file
    utils.table_creation(['R^2', 'RMSE', 'MAE'], [r2, rmse, mae], 'kernel_ridge_regression_gaussian.txt')

    """
    # Both plots
    ax = pred.plot(x='date', y='RefSt')
    ax2 = pred.plot(x='date', y='Kernel_Gauss_Pred', ax=ax)
    pred.plot(x='date', y='Best_Kernel_Poly_Pred', ax=ax,
                   title='Different')
    plt.show()
    # plt.savefig(path_kernel_ridge_regression_plots + str("models/kernel_gauss_poly_model.png"), bbox_inches='tight')
    # plt.clf()
    """

