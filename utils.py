import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


def normalize(data):
    norm = (data - data.mean()) / data.std()
    return norm


# print mean of each column a the data frame
def get_means(data):
    means = data.mean()
    return means


# print covariance matrix a the data frame
def get_cov_matrix(data):
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    cov = data.cov()
    return cov


# create train and test sets
def train_test_creation(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, shuffle=False)
    return x_train, x_test, y_train, y_test


# compute best features
def forward_subset_selection(x, y, k):
    # Check if missing values for the data
    lreg = LinearRegression()

    # FORWARD by R2
    sfs_r2 = sfs(lreg, k_features=k, forward=True, scoring='r2').fit(x, y)
    sfs_mae = sfs(lreg, k_features=k, forward=True, scoring='neg_mean_absolute_error').fit(x, y)
    sfs_mse = sfs(lreg, k_features=k, forward=True, scoring='neg_mean_squared_error').fit(x, y)

    feat_names = list(sfs_r2.k_feature_names_)
    return feat_names  # Return the best features


def loss_functions(y_pred, y_true):
    print("Loss functions:")
    print("* R-squared =", r2_score(y_true, y_pred))
    print("* RMSE =", mean_squared_error(y_true, y_pred))
    print("* MAE =", mean_absolute_error(y_true, y_pred))