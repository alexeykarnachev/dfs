import sys
sys.path.append('../src/')

from sklearn.model_selection import KFold
from itertools import product
from dfs.DFSModel import DFSModel
import numpy as np
import os

from dfs.utils import read_insilico_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_FILE = '../data/train.csv'
SUMMARY_FILE = '../summary/train_dfs.csv'
K_FOLDS = 10
NUM_EPOCHS = 300
BATCH_SIZE = 128

if __name__ == '__main__':
    x, y = read_insilico_data(DATA_FILE)

    kf = KFold(n_splits=K_FOLDS, shuffle=True)

    input_dim_ = [x.shape[1]]
    output_dim_ = [y.shape[1]]
    hidden_dims_ = [[50], [100], [300], [1000], [50, 50], [100, 100], [300, 300], [1000, 1000], [50, 50, 50],
                    [100, 100, 100]]
    lambda_1_ = [0.0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.2]
    lambda_2_ = [1.0]
    alpha_1_ = [0.0001]
    alpha_2_ = [0.0]
    lr_ = [0.0001, 0.001]

    summary_columns = ['input_dim', 'output_dim', 'hidden_dims', 'lambda_1', 'lambda_2', 'alpha_1', 'alpha_2', 'lr']
    summary_columns += ['train_loss_mean', 'train_roc_auc_mean', 'test_loss_mean', 'test_roc_auc_mean', 'n_features_mean',
                        'true_y_mean']
    summary_columns += ['train_loss_std', 'train_roc_auc_std', 'test_loss_std', 'test_roc_auc_std', 'n_features_std',
                        'true_y_std']

    with open(SUMMARY_FILE, 'w') as f:
        f.write(','.join(summary_columns) + '\n')

    params_iter = list(product(input_dim_, output_dim_, hidden_dims_, lambda_1_, lambda_2_, alpha_1_, alpha_2_, lr_))
    np.random.shuffle(params_iter)
    print('Total number of param combinations: {}'.format(len(params_iter)))

    for i_params in range(len(params_iter)):
        params = params_iter[i_params]
        params_str = '_'.join([str(x) for x in params])

        i_fold = 0
        history = []
        for train_index, test_index in kf.split(x):
            x_train, y_train = x[train_index], y[train_index]
            validation_data = (x[test_index], y[test_index])
            true_y_mean = y_train.mean()

            model = DFSModel(*params)

            print('\n\n\nParams Set: {}/{}\tFold: {}/{}\t Params: {}'.format(i_params + 1, len(params_iter), i_fold + 1,
                                                                             K_FOLDS, params))
            model.fit(x=x_train, y=y_train, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=validation_data, verbose=1)

            i_fold += 1

            features_weights = model.features_weights
            n_features = sum(np.abs(features_weights) > model.F_EPS)

            history.append(
                [model.train_loss, model.train_roc_auc, model.test_loss, model.test_roc_auc, n_features, true_y_mean])

        history = np.vstack(history)

        summary_values = list(params) + list(history.mean(axis=0)) + list(history.std(axis=0))

        with open(SUMMARY_FILE, 'a') as f:
            f.write(','.join([str(x) for x in summary_values]) + '\n')

        print(dict(zip(summary_columns, summary_values)))
