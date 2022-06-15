import pandas as pd
import data


# print mean of each metric
def get_means():
    means = data.new_PR_data_inner.mean()
    return means


def get_cov_matrix():
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    cov = data.new_PR_data_inner.cov()
    return cov
