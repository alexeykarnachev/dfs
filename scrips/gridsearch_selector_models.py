import sys
sys.path.append('../src/')

from dfs.utils import *
import json
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = '../data/train.csv'
PARAMS_FILE = '../summary/fs_skl_models.json'
K_FOLDS = 10


if __name__ == '__main__':
    params = {}
    x, y = read_insilico_data(DATA_FILE)
    y = y[:, 0].astype(int)
    kf = KFold(n_splits=K_FOLDS, shuffle=True)

    # ==================================================================================================================
    # XGBClassifier:
    parameters = {'n_estimators': [30, 100, 300, 600],
                  'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.99, 2.0],
                  'max_depth': [2, 4, 8, 12]}

    xgb = GridSearchCV(XGBClassifier(), parameters, cv=kf, n_jobs=10, verbose=10, scoring='roc_auc')
    xgb.fit(x, y)

    # ==================================================================================================================
    # SGDClassifier:
    parameters = {'alpha': [0.0001, 0.01, 0.05, 0.1, 0.5, 1.0],
                  'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.99]}

    els = GridSearchCV(SGDClassifier(penalty='elasticnet'), parameters, cv=kf, n_jobs=10, verbose=10, scoring='roc_auc')
    els.fit(x, y)

    # ==================================================================================================================
    # RandomForestClassifier:
    parameters = {'n_estimators': [30, 100, 300, 600],
                  'max_depth': [2, 4, 8, 12],
                  'min_samples_leaf': [1, 5, 20],
                  'min_samples_split': [2, 10, 40]}

    rdf = GridSearchCV(RandomForestClassifier(), parameters, cv=kf, n_jobs=10, verbose=10, scoring='roc_auc')
    rdf.fit(x, y)

    # ==================================================================================================================
    # Save parameters:
    params['XGBClassifier'] = xgb.best_estimator_.get_params()
    params['SGDClassifier'] = els.best_estimator_.get_params()
    params['RandomForestClassifier'] = rdf.best_estimator_.get_params()

    with open(PARAMS_FILE, 'w') as f:
        json.dump(params, f)