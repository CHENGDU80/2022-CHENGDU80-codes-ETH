import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, plot_roc_curve
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from preprocess import train_prep_pca, eval_prep_pca
from util import *
import pickle
import time
import argparse
from model import load_model

def save_model_pca(model, features, pca_g0, pca_gl0, pca_int, save_name = ''):
    new_dict = {'model': model, 'features': features, 'pca_g0': pca_g0, 'pca_gl0':pca_gl0, 'pca_int':pca_int}
    filename = 'model/model_'+ save_name + str(int(time.time()))+'.pkl'
    pickle.dump(new_dict, open(filename, 'wb'))

def train_pca(model, fs_method, fs_model, save_name = '', over_sample = False, fs = 1):
    # pipeline
    X_train, Y_train, pca_g0, pca_gl0, pca_int = train_prep_pca()
    print('training data preprocessed')
    features, X_train_select = FeatureSelection(X_train, Y_train, fs_model, fs_method)
    print('selected features', features)
    print(len(features))
    if fs == 2:
        features, X_train_select = FeatureSelection(X_train_select, Y_train, fs_model, fs_method)
        print('selected features', features)
        print(len(features))
    if over_sample == True:
        X, Y = oversample(0.8, X_train_select, Y_train[['DEFAULT_LABEL']])
    else:
        X, Y = X_train_select, Y_train[['DEFAULT_LABEL']]
    model.fit(X, Y['DEFAULT_LABEL'])
    train_score = model.predict_proba(X_train_select)
    print('train auc', roc_auc_score(Y_train['DEFAULT_LABEL'], train_score[:, 1]))
    save_model_pca(model, features, pca_g0, pca_gl0, pca_int, save_name)
    plot_roc_curve(model, X_train_select, Y_train['DEFAULT_LABEL'])
    return features, model

def evaluate_pca(filename):
    loaded_model = load_model('model/'+filename + '.pkl')
    model = loaded_model['model']
    features = loaded_model['features']
    pca_g0 = loaded_model['pca_g0']
    pca_gl0 = loaded_model['pca_gl0']
    pca_int = loaded_model['pca_int']

    X_test, Y_test = eval_prep_pca(pca_g0, pca_gl0, pca_int)
    X_test_select = X_test[features]
    test_score = model.predict_proba(X_test_select)
    print('eval auc', roc_auc_score(Y_test['DEFAULT_LABEL'], test_score[:, 1]))
    plot_roc_curve(model, X_test_select, Y_test['DEFAULT_LABEL'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='evaluate', help='train, evaluate, test')
    parser.add_argument('--model_name', default='pca_gb_brf', help='some packaged model name')
    parser.add_argument('--modelfile_name', default=None, help='the name of the model file')



    args = parser.parse_args()

    ############## training stage ###############
    if args.stage == 'train':
        if args.model_name == 'pca_brf_lasso':
            # scaling ('quantile', 'standard')
            scaling = 'quantile'
            # na filling method (-10.0 -500.0 'mean' 0)
            nafill = -10.0
            # features selection method ('lasso' 'rf')
            fs = 1
            fs_method = 'lasso'
            fs_model = Lasso(alpha = 0.0001)
            # over sample
            over_sample = False
            # model (brf gb)
            model = BalancedRandomForestClassifier(n_estimators=500, random_state = 0, bootstrap = False, sampling_strategy = 0.8, max_samples = 3000)
            
            
            
        if args.model_name == 'pca_brf_brf':
            # scaling
            scaling = 'standard'
            # na filling method
            nafill = -500.0
            # model
            model = BalancedRandomForestClassifier(n_estimators=500, random_state = 0, bootstrap = False, sampling_strategy = 0.8, max_samples = 3000)
            over_sample = False
            # features selection method
            fs = 1
            fs_method = 'rf'
            fs_model = BalancedRandomForestClassifier(n_estimators=100, random_state = 0, bootstrap = False, max_samples = 10000)

        if args.model_name == 'pca_gb_brf':
            # scaling
            scaling = 'standard'
            # na filling method
            nafill = -500.0
            # model
            model = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, subsample = 0.4, n_estimators=100, max_depth=3, random_state=0, validation_fraction=0.1)
            over_sample = True
            # features selection method
            fs = 1
            fs_method = 'rf'
            fs_model = BalancedRandomForestClassifier(n_estimators=100, random_state = 0, bootstrap = False, max_samples = 10000)

        if args.model_name == 'pca_gb_brf_2':
            # scaling
            scaling = 'standard'
            # na filling method
            nafill = -500.0
            # model
            model = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, subsample = 0.4, n_estimators=100, max_depth=3, random_state=0, validation_fraction=0.1)
            over_sample = True
            # features selection method
            fs = 2
            fs_method = 'rf'
            fs_model = BalancedRandomForestClassifier(n_estimators=100, random_state = 0, bootstrap = False, max_samples = 10000)
            


        train_pca(model, fs_method, fs_model, args.model_name, over_sample, fs)


    ################# evaluation stage #################
    if args.stage == 'evaluate':
        evaluate_pca(args.modelfile_name)

    # ################# test stage #####################
    # if args.stage == 'test':
    #     test(args.modelfile_name)
