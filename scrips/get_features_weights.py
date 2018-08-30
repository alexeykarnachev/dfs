import sys
sys.path.append('../src/')

import json
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import inspect
from dfs.DFSModel import DFSModel
from dfs.utils import *
import pandas as pd
from sklearn.metrics import roc_auc_score

DFS_REPORT = '../summary/train_dfs.csv'
DATA_FILE = '../data/train.csv'
SKL_PARAMS = '../summary/fs_skl_models.json'
FEATURES_REPORT = '../summary/features_weights.json'
K_FOLDS = 10


if __name__ == '__main__':
    x, y = read_insilico_data(DATA_FILE)
    kf = KFold(n_splits=K_FOLDS, shuffle=True)

    dfs_report = read_dfs_report(DFS_REPORT)
    skl_params = read_json(SKL_PARAMS)
    REPORT = {}

    dfs_params = dfs_report.loc[53].to_dict()
    dfs_params = {x: dfs_params[x] for x in list(inspect.signature(DFSModel).parameters.keys())}

    # ==================================================================================================================
    # DFSModel:
    test_acc = []
    features = []

    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        validation_data = (x[test_index], y[test_index])
        dfs = DFSModel(**dfs_params)
        dfs.fit(x=x_train, y=y_train, num_epochs=300, batch_size=128, validation_data=validation_data, verbose=2)
        test_acc.append(roc_auc_score(y[test_index], dfs.predict(x[test_index])))
        features.append(dfs.get_feature_weights())

    test_acc = np.mean(test_acc)
    features = pd.Series(np.abs(np.vstack(features)).mean(axis=0)).sort_values(ascending=False)
    REPORT['DFSModel'] = {'kfold_roc_auc': test_acc, 'sorted_features': features.index.values,
                          'feature_weights': features.values}

    # ==================================================================================================================
    # XGBClassifier:
    test_acc = []
    features = []

    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        xgb = XGBClassifier(**skl_params['XGBClassifier'])
        xgb.fit(X=x_train, y=y_train[:, 0].astype(int))
        test_acc.append(xgb.score(x_test, y_test))
        features.append(xgb.feature_importances_)

    test_acc = np.mean(test_acc)
    features = pd.Series(np.abs(np.vstack(features)).mean(axis=0)).sort_values(ascending=False)
    REPORT['XGBClassifier'] = {'kfold_roc_auc': test_acc, 'sorted_features': features.index.values,
                               'feature_weights': features.values}

    # ==================================================================================================================
    # SGDClassifier:
    test_acc = []
    features = []

    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        els = SGDClassifier(**skl_params['SGDClassifier'])
        els.fit(X=x_train, y=y_train[:, 0].astype(int))
        test_acc.append(els.score(x_test, y_test))
        features.append(els.coef_)

    test_acc = np.mean(test_acc)
    features = pd.Series(np.abs(np.vstack(features)).mean(axis=0)).sort_values(ascending=False)
    REPORT['SGDClassifier'] = {'kfold_roc_auc': test_acc, 'sorted_features': features.index.values,
                               'feature_weights': features.values}

    # ==================================================================================================================
    # RandomForestClassifier:
    test_acc = []
    features = []

    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        rdf = RandomForestClassifier(**skl_params['RandomForestClassifier'])
        rdf.fit(X=x_train, y=y_train[:, 0].astype(int))
        test_acc.append(rdf.score(x_test, y_test))
        features.append(rdf.feature_importances_)

    test_acc = np.mean(test_acc)
    features = pd.Series(np.vstack(features).mean(axis=0)).sort_values(ascending=False)
    REPORT['RandomForestClassifier'] = {'kfold_roc_auc': test_acc, 'sorted_features': features.index.values,
                                        'feature_weights': features.values}

    # ==================================================================================================================
    # Save Report:
    with open(FEATURES_REPORT, 'w') as f:
        json.dump(REPORT, f, cls=NumpyEncoder)