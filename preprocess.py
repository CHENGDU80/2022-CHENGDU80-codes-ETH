import pandas as pd
from util import *
import pickle

def train_prep(scale_method, fillna_m):
    train_data = pd.read_csv('../data/train/feature.csv', index_col=0)
    train_label = pd.read_csv('../data/train/label.csv', index_col = 0)
    print('data loaded')
    train_data = na_rate(train_data)
    train_data, train_label = datediff(train_data, train_label)
    print('feature generated')
    train_scale,  std_scaler = scale_std(train_data, name = scale_method)
    train_fillna, fillm= fillna_method(train_scale, method = fillna_m)
    print('scaled and na filled')

    return train_fillna, train_label, std_scaler, fillm

def eval_prep(std_scaler, fillna_m):
    test_data = pd.read_csv('../data/test/feature.csv', index_col = 0)
    test_label = pd.read_csv('../data/test/label.csv', index_col = 0)
    test_data = na_rate(test_data)
    test_data, test_label = datediff(test_data, test_label)

    test_scale = pd.DataFrame(std_scaler.transform(test_data), columns = test_data.columns)
    test_fillna, _ = fillna_method(test_scale, method = fillna_m)
    return test_fillna, test_label

def test_prep(std_scaler, fillna_m):
    test_data = pd.read_csv('../data/final/feature.csv', index_col = 0)
    test_sample = pd.read_csv('../data/final/sample.csv', index_col = 0)
    test_data = na_rate(test_data)
    test_data, _= datediff(test_data, test_sample)

    test_scale = pd.DataFrame(std_scaler.transform(test_data), columns = test_data.columns)
    test_fillna, _ = fillna_method(test_scale, method = fillna_m)
    return test_fillna, test_sample


def avg_combine(X):
    feature_dict = pickle.load(open('feature_columns.pkl', 'rb'))
    X_avg_g0 = avg_data(X[feature_dict['float_g0']])
    X_avg_gl0 = avg_data(X[feature_dict['float_gl0']])
    X_avg_g0.columns = ['g0_'+col for col in X_avg_g0.columns]
    X_avg_gl0.columns = ['gl0_'+col for col in X_avg_gl0.columns]

    X_float_l0 = X[feature_dict['float_l0']]
    X_int = X[feature_dict['int_feature']]
    X_int = X_int.fillna(-1)

    X_new = pd.concat([X_avg_g0, X_avg_gl0, X_float_l0, X_int], axis = 1)
    X_new['na_rate'] = X['na_rate']
    return X_new

def train_prep_avg():
    train_data = pd.read_csv('../data/train/feature.csv', index_col=0)
    train_label = pd.read_csv('../data/train/label.csv', index_col = 0)
    print('data loaded')
    train_data = na_rate(train_data)
    train_data, train_label = datediff(train_data, train_label)
    print('feature generated')
    train_new = avg_combine(train_data)
    print('data averaged and combined')
    return train_new, train_label

def eval_prep_avg():
    test_data = pd.read_csv('../data/test/feature.csv', index_col = 0)
    test_label = pd.read_csv('../data/test/label.csv', index_col = 0)
    test_data = na_rate(test_data)
    test_data, test_label = datediff(test_data, test_label)
    test_new = avg_combine(test_data)
    return test_new, test_label