import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

from dfs.utils import chunks


class DFSModel:
    EPS = 1e-7
    F_EPS = 5e-5

    def __init__(self, input_dim, output_dim, hidden_dims, lambda_1, lambda_2, alpha_1, alpha_2, lr):
        tf.reset_default_graph()
        self.input_dim = np.int32(input_dim)
        self.hidden_dims = hidden_dims
        self.output_dim = np.int32(output_dim)
        self.lambda_1 = np.float32(lambda_1)
        self.lambda_2 = np.float32(lambda_2)
        self.alpha_1 = np.float32(alpha_1)
        self.alpha_2 = np.float32(alpha_2)
        self.lr = np.float32(lr)
        self.sess = None
        self._build_model()

    def _build_model(self):
        with tf.variable_scope('input'):
            self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_dim), name='x_input')
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y_input')

        with tf.variable_scope('network'):
            hadamard_layer = self.get_hadamard_layer('hadamard_layer', x=self.x)
            network_layers = [hadamard_layer]
            num_neurons_layers = self.hidden_dims + [self.output_dim]
            n_layers = len(num_neurons_layers)
            for i_layer in range(n_layers):
                num_neurons = num_neurons_layers[i_layer]
                name = 'layer_{}'.format(i_layer) if i_layer < n_layers else 'layer_out'.format(i_layer)
                previous_layer = network_layers[-1]
                network_layers.append(self.get_dense_layer(name=name, x=previous_layer, num_neurons=num_neurons,
                                                           activation=tf.nn.sigmoid))

        with tf.variable_scope('loss'):
            log_loss = self.get_binary_crossentropy_loss('log_loss', y_true=self.y, y_pred=network_layers[-1])
            elastic_inp_loss = self.get_elasticnet_loss('elastic_inp_loss', layers=[network_layers[0]],
                                                        lambda_1=self.lambda_1, lambda_2=self.lambda_2)
            elastic_hid_loss = self.get_elasticnet_loss('elastic_hid_loss', layers=[network_layers[1:]],
                                                        lambda_1=self.alpha_1, lambda_2=self.alpha_2)
            self.loss = log_loss + elastic_inp_loss + elastic_hid_loss

        with tf.variable_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)

        self.__y_pred = network_layers[-1]
        self.__feature_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'network/hadamard_layer/W')[0]

    @staticmethod
    def get_dense_layer(name, x, num_neurons, activation):
        with tf.variable_scope(name):
            shape = (x.get_shape()[1], num_neurons)
            w = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable("b", shape=num_neurons, initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            z = tf.matmul(x, w) + b
            a = activation(z)
            return a

    @staticmethod
    def get_hadamard_layer(name, x):
        with tf.variable_scope(name):
            w = tf.get_variable("W", shape=x.shape[1], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            z = tf.multiply(x, w)
            return z

    @staticmethod
    def get_binary_crossentropy_loss(name, y_true, y_pred):
        with tf.variable_scope(name):
            y_pred = tf.clip_by_value(y_pred, clip_value_min=DFSModel.EPS, clip_value_max=1 - DFSModel.EPS)
            l = -1 * tf.reduce_mean(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred), name='loss')
            return l

    @staticmethod
    def get_elasticnet_loss(name, layers, lambda_1, lambda_2):

        weights = []
        for layer in layers:
            layer_name = layer.name.split('/')[1]
            weights_name = 'network/{}/W:0'.format(layer_name)
            weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, weights_name)[0])

        with tf.variable_scope(name):
            l1_loss = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in weights])
            l2_loss = tf.reduce_sum([tf.reduce_sum(w ** 2) for w in weights])
            lambda_k = (1 - lambda_2) / 2
            l = tf.multiply(lambda_1, (lambda_k * l2_loss + lambda_2 * l1_loss), name='loss')
            return l

    def init_sess(self):
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def fit(self, x, y, num_epochs, batch_size, validation_data, verbose):
        if self.sess is None:
            self.init_sess()

        x_test, y_test = validation_data

        for i_epoch in range(num_epochs):
            inds = chunks(list(range(len(x))), batch_size)

            for batch_inds in inds:
                self.sess.run(self.train_step, {self.x: x[batch_inds], self.y: y[batch_inds]})

            train_loss = self.sess.run(self.loss, {self.x: x, self.y: y})
            test_loss = self.sess.run(self.loss, {self.x: x_test, self.y: y_test})

            train_roc_auc = roc_auc_score(y, self.predict(x))
            test_roc_auc = roc_auc_score(y_test, self.predict(x_test))
            mean_roc_auc = roc_auc_score(y_test, [np.mean(y_test)] * len(y_test))

            self.features_weights = self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                    'network/hadamard_layer/W')[0])
            n_features = sum(np.abs(self.features_weights) > DFSModel.F_EPS)

            if verbose >= 2:
                self._verbose(i_epoch, num_epochs, train_loss, train_roc_auc, test_loss, test_roc_auc, mean_roc_auc,
                              n_features)

        if verbose >= 1:
            self._verbose(i_epoch, num_epochs, train_loss, train_roc_auc, test_loss, test_roc_auc, mean_roc_auc, n_features)

        self.train_loss, self.train_roc_auc, self.test_loss, self.test_roc_auc = train_loss, train_roc_auc, test_loss, test_roc_auc

    def predict(self, x):
        if self.sess is None:
            self.init_sess()

        return self.sess.run(self.__y_pred, {self.x: x})

    def get_feature_weights(self):
        if self.sess is None:
            self.init_sess()

        return self.sess.run(self.__feature_weights)

    def _verbose(self, i_epoch, num_epochs, train_loss, train_roc_auc, test_loss, test_roc_auc, mean_roc_auc, n_features):
        print('Epoch: {}/{}\ttrain_loss:{:.3f}\ttrain_roc_auc:{:.3f}\t'
              'test_loss:{:.3f}\ttest_roc_auc:{:.3f}\tmean_roc_auc:{:.3f}\tn_features:{}'.
              format(i_epoch + 1, num_epochs, train_loss, train_roc_auc, test_loss, test_roc_auc, mean_roc_auc, n_features))
