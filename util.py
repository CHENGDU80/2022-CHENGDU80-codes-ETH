import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA


def filter_with_na(X, Y, X_test, row_thres = 800, col_thres = 73500):
    # column na count
    na_col_count = []
    for i in range(len(X.columns)):
        na_col_count.append(sum(pd.isna(X.iloc[:, i])))
    na_col_count = np.array(na_col_count)
    # row na count
    na_row_count = []
    for i in range(len(X)):
        na_row_count.append(sum(pd.isna(X.iloc[i, :])))
    na_row_count = np.array(na_row_count)
    X = X.loc[na_row_count <= row_thres, na_col_count <= col_thres]
    Y = Y.loc[na_row_count <= row_thres]
    X_test = X_test.loc[:, na_col_count <= col_thres]
    return X, Y, X_test


def fillna_method(X, method = None):
    if method == "knn":
        imputer = KNNImputer(weights = "distance")
        return pd.DataFrame(
        imputer.fit_transform(X),
        columns = X.columns)
    elif method == 'mean':
        X_new = X.fillna(X.mean())
        return X_new, X.mean()
    elif method == '0':
        X_new = X.fillna(0)
    else:
        X_new = X.fillna(method)
    return X_new, method


def scale_std(X,name = 'quantile'):
    if name == 'quantile':
        std_scaler = QuantileTransformer(output_distribution="normal", random_state = 0).fit(X)
    if name == 'standard':
        std_scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(std_scaler.transform(X), columns = X.columns)
    return X_scaled, std_scaler


def oversample(strategy, x, y):
    return RandomOverSampler(sampling_strategy=strategy, random_state=0).fit_resample(x, y)


def undersample(strategy, x, y):
    return RandomUnderSampler(sampling_strategy=strategy, random_state=0).fit_resample(x, y)

def na_rate(X):
    # na_rate
    X['na_rate'] = X.isna().sum(axis = 1)/X.shape[1]
    return X

def datediff(X, Y):
    date_start = datetime.strptime('2021-05-19', '%Y-%m-%d')
    Y['date'] = Y.apply(lambda x:datetime.strptime(x['APPLICATION_DATE'], '%Y-%m-%d'), axis = 1)
    X['datediff'] = (Y['date'] - date_start).dt.days
    return X, Y

def FeatureSelection(X, Y, model, method = 'rf', alpha = 0.01):
    if method == 'rf':
        sfm = SelectFromModel(model, threshold = "mean")
        sfm.fit(X, Y['DEFAULT_LABEL'])
        features = X.columns[sfm.get_support(indices = True)]
        X_train_select = X[features]
    elif method == 'lasso':
        model.fit(X, Y['DEFAULT_LABEL'])
        features = X.columns[model.coef_>0]
        X_train_select = X[features]
    return features, X_train_select

def avg_data(data):
    columns_data = len(data.columns)
    avg_idx = list(np.arange(0, columns_data+1, 12))
    if avg_idx[-1] != columns_data:
        avg_idx.append(columns_data)
    avg_columns = ['avg'+str(i) for i in range(len(avg_idx)-1)]
    data_new = pd.DataFrame(columns = avg_columns)
    for i in range(len(avg_idx)-1):
        data_new.loc[:, avg_columns[i]] = data.iloc[:, avg_idx[i]:avg_idx[i+1]].mean(axis = 1)
    return data_new

def PCA_transform(X, n):
    pca = PCA(n_components=n, random_state=0)
    pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    X_pca = pd.DataFrame(pca.transform(X), columns = ['pca'+str(i) for i in range(n)])
    return X_pca, pca

