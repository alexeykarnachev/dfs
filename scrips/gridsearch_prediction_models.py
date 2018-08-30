import json
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from dfs.utils import *

FEATURES_WEIGHTS = '../summary/features_weights.json'
FINAL_REPORT = '../summary/final_report.json'
DATA_FILE = '../data/train.csv'
K_FOLDS = 10
MAX_N_FEATURES = 200

with open(FEATURES_WEIGHTS) as f:
    features_weights = json.load(f)

if __name__ == '__main__':

    x, y = read_insilico_data(DATA_FILE)
    y = y[:, 0].astype(int)

    kf = KFold(n_splits=K_FOLDS, shuffle=True)
    models = {KNeighborsClassifier: {'n_neighbors': [2, 5, 20]},
              XGBClassifier: {'n_estimators': [100],
                              'learning_rate': [0.01, 0.3, 0.99],
                              'max_depth': [2, 4, 8]},
              RandomForestClassifier: {'n_estimators': [100],
                                       'max_depth': [2, 4, 8],
                                       'min_samples_leaf': [1, 20],
                                       'min_samples_split': [2, 40]},
              LogisticRegression: dict()}

    REPORT = {}

    for features_model in features_weights:
        features_model_roc_auc = features_weights[features_model]['kfold_roc_auc']
        sorted_features = features_weights[features_model]['sorted_features']
        REPORT[features_model] = {}

        for model in models:
            prediction_model = model.__name__
            REPORT[features_model][prediction_model] = {'n_features': [], 'score': []}

            for n_features in range(1, MAX_N_FEATURES + 1):
                features_to_use = sorted_features[:n_features]
                search_params = models[model]
                clf = GridSearchCV(model(), search_params, cv=kf, n_jobs=12, verbose=0, scoring='roc_auc')
                clf.fit(x[:, features_to_use], y)

                REPORT[features_model][prediction_model]['n_features'].append(n_features)
                REPORT[features_model][prediction_model]['score'].append(clf.best_score_)

                print(features_model, prediction_model, n_features, clf.best_score_)

    with open(FINAL_REPORT, 'w') as f:
        json.dump(REPORT, f, cls=NumpyEncoder)
