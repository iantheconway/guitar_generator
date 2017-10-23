from tensorflow.python.client import device_lib

import argparse
import os
import logging
import pickle
import json
import sys
import time

PLATFORM = sys.platform

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.contrib import rnn
import scipy.io.wavfile


# set tensorflow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


'''
Code to generate raw audio trained on guitar
by Ian Conway
'''


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
np.random.seed(42)
tf.set_random_seed(42)


# arg parser - No arguments are required
def build_arg_parser():
    parser = argparse.ArgumentParser(description='This code does static and dynamic'
                                                 ' forecasting on generated'
                                                 ' data using LSTM Neural Networks')
    parser.add_argument("-c", help="model configuration name")
    parser.add_argument("-g", help="model grid config name")
    return parser


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class LSTMForecasting(object):
    def __init__(self):
        # Scaling parameters
        val_map = pickle.load(open(os.path.join(CURRENT_DIR, "scaling.p"), "rb"))
        self.mean = val_map["mean"]
        self.std = val_map["std"]

        # split point for training/test
        self.cutoff = -1
        self.dynamic_cutoff = -200

        # Use artificial forecasts
        self.include_fauxcast = True

        # TensorFlow Parameters
        self.training = False
        self.training_cycles = 1
        self.learning_rate = 0.001
        self.training_iters = 200000
        self.batch_size = 24
        self.display_step = 2
        self.save_step = 20
        self.n_input = 1  # number of input features
        self.sample_rate = 16000
        self.n_steps = int(self.sample_rate / 2)
        self.n_hidden = 64  # number of hidden layer units
        self.n_classes = 1  # No longer a class, as we are doing regression
        self.output_steps = 1
        self.time_delta = 1
        self.num_layers = 3
        self.regularization = .00  # use dropout instead
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        self.keep_prob_param = .5
        self.cell_type = "lstm"
        self.sess = tf.Session()  # initiate TensorFlow attributes

        gpus = get_available_gpus()

        if len(gpus) != 0:
            self.device = gpus[0]
        else:
            local_device_protos = device_lib.list_local_devices()
            self.device = local_device_protos[0].name

        self.experiment_name = 'default'
        self.model_dir = os.path.join("models", self.experiment_name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.grid_mode = False
        self.best_error = 9999.9


    def load_config(self, config_experiment):
        if isinstance(config_experiment, dict):
            json_config = config_experiment
            config_experiment = json_config['experiment_name']
        else:
            if config_experiment[-5:] == '.json'.lower():
                config_experiment = config_experiment[:-5]

            # Open config file
            with open('{}.json'.format(config_experiment)) as json_str:
                json_config = json.load(json_str)

        input_params = json_config['input_params']
        self.experiment_name = json_config['experiment_name']
        self.model_dir = os.path.join("models", self.experiment_name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.json_config = json_config

        # Load input params
        print('Load inputs parameters')
        for param in input_params:
            if param in ['batch_size', 'n_steps', 'n_hidden', 'num_layers']:
                input_params[param] = int(input_params[param])

            setattr(self, param, input_params[param])
            print(param, input_params[param])

        # Create model folder according to experiment_name
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.image_dir)
        else:
            print('Experiment folder already exists, creating new folder.')
            i_name = 0
            model_dir = os.path.join("models", '{n}_{i}'.format(n=self.experiment_name, i=i_name))
            while os.path.exists(model_dir):
                i_name += 1
                model_dir = os.path.join("models", '{n}_{i}'.format(n=self.experiment_name, i=i_name))

            self.experiment_name = '{n}_{i}'.format(n=self.experiment_name, i=i_name)
            self.model_dir = os.path.join("models", self.experiment_name)
            self.image_dir = os.path.join(self.model_dir, 'images')
            os.makedirs(self.model_dir)
            os.makedirs(self.image_dir)

        # Create initial results csv
        self.results_csv = os.path.join(self.model_dir, '_results_{}.csv'.format(self.experiment_name))
        with open(self.results_csv, 'wb') as f:
            f.write('prediction_time,total_iterations,mse_train,mse_test\n')

        # Save config file
        with open(os.path.join(self.model_dir, '{}.json'.format(config_experiment)), 'w') as json_file:
            json.dump(self.json_config, json_file)

    def load_features(self):

        features = scipy.io.wavfile.read("./data/guitar.wav")
        print features
        self.sample_rate = features[0]
        features = np.array(features[1])
        print features
        labels = features.copy()[self.n_steps:]
        features = features[:-self.n_steps]

        self.features_pure = np.copy(features)

        std = np.std(features, axis=0)
        mean = np.mean(features, axis=0)
        vals = {"mean": mean, "std": std}
        pickle.dump(vals, open("scaling.p", "wb"))
        features = features - mean
        features = np.divide(features, std)

        self.features_pure = self.features_pure - mean
        self.features_pure = np.divide(self.features_pure, std)

        self.cutoff = int((features.shape[0]) * .8)
        self.features_train = features[:self.cutoff]
        self.features_test = features[self.cutoff:]
        self.labels_train = features[:self.cutoff]
        self.labels_test = features[self.cutoff:]
        self.feature_vals = vals
        print("Parsed {} values from csv".format(len(features)))

    # return a randomly selected batch of x and y values
    def get_batch(self):
        window_starts = np.random.randint(0,
                                          high=self.cutoff - self.n_steps -
                                               (self.output_steps - 1) - 1 - (self.time_delta - 1),
                                          size=[self.batch_size])
        x_batch = []
        for i in window_starts:
            vector = []
            for j in range(self.n_steps):
                vector.append(self.features_train[i + j])

            x_batch.append(vector)

        x_batch = np.array(x_batch)
        y_batch = np.array([self.labels_train[
                            i + self.n_steps + (self.time_delta - 1):
                            i + self.n_steps + self.output_steps + (self.time_delta - 1)] for i in window_starts])

        return x_batch, y_batch

    def tf_graph(self, name=''):
        # tf Graph input
        print("initializing tf graph")
        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes * self.output_steps])
        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes * self.output_steps]),
                               name="weights")
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes, self.output_steps]), name="biases")
        }

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        self.x_unstacked = tf.unstack(self.x, self.batch_size, 0)

        # Define a cell with tensorflow
        if self.cell_type == "gru":
            print("using GRU")
            self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(self.n_hidden),
                output_keep_prob=self.keep_prob) for _ in range(self.num_layers)], state_is_tuple=True)

        elif self.cell_type == "lstm":
            self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0),
                output_keep_prob=self.keep_prob) for _ in range(self.num_layers)], state_is_tuple=True)

        elif self.cell_type == "ugrnn":
            self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                rnn.UGRNNCell(self.n_hidden),
                output_keep_prob=self.keep_prob) for _ in range(self.num_layers)], state_is_tuple=True)

        elif self.cell_type == "nas":
            self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.NASCell(self.n_hidden),
                output_keep_prob=self.keep_prob) for _ in range(self.num_layers)], state_is_tuple=True)

        elif self.cell_type == "lnb":
            print "!"
            self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hidden),
                output_keep_prob=self.keep_prob) for _ in range(self.num_layers)], state_is_tuple=True)

        else:
            print "ERROR: Bad cell type or cell type not defined"
            exit()
        # Get cell output
        self.outputs, self.states = tf.nn.dynamic_rnn(self.cell, self.x, dtype=tf.float32)
        self.outputs = self.outputs[:,-1,:]
        # self.outputs = tf.reshape(self.outputs, [self.batch_size, -1, self.n_hidden])
        # Linear activation, using rnn inner loop last output
        self.outputs_unstacked = tf.unstack(self.outputs, self.batch_size)

        self.pred = tf.add(tf.matmul(self.outputs, self.weights['out'], name="multiply_out_weights"),
                           self.biases['out'],
                           name="add_output_bias")
        print self.pred

        self.average_error = tf.reduce_mean(tf.sqrt(tf.clip_by_value(
            tf.squared_difference(self.pred, self.y), 1e-37, 1e+37)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.average_error)
        # Initializing the variables
        init = tf.global_variables_initializer()

        # for tensorboard

        tf.summary.scalar("average_error", self.average_error)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs/" + str(name))
        self.writer.add_graph(self.sess.graph)
        # Launch the graph
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def test(self, num=1):
        # Run predictions on the test data and plot the results
        test_data = self.features_test
        test_labels = np.array(self.features_test[self.n_steps + (self.time_delta - 1):])
        # test_labels = test_labels.reshape((test_labels.shape[0]/self.output_steps, self.output_steps))
        i = 0
        j = self.n_steps
        self.data = []
        while j <= len(test_data) - (self.time_delta - 1):
            self.data.append(test_data[i:j])
            i += self.output_steps
            j += self.output_steps

        self.data = np.array(self.data)
        print("Testing set size: " + str(len(test_labels)))
        self.data = np.reshape(self.data, (self.data.shape[0], self.n_steps, self.n_input))

        train_data = self.features_train
        train_labels = np.array(self.features_train[self.n_steps:])

        i = 0
        j = self.n_steps
        train = []
        while j <= len(train_data):
            train.append(train_data[i:j])
            i += self.output_steps
            j += self.output_steps

        train = np.array(train)
        train = np.reshape(train, (train.shape[0], self.n_steps, self.n_input))

        label_std = self.feature_vals['std'][0]
        label_mean = self.feature_vals['mean'][0]

        self.preds = self.sess.run(self.pred, feed_dict={self.x: self.data, self.keep_prob: 1.})
        self.preds = self.preds.flatten()
        preds_train = self.sess.run(self.pred, feed_dict={self.x: train, self.keep_prob: 1.})
        preds_train = preds_train.flatten()

        preds_unscaled = self.preds * label_std + label_mean
        preds_train_unscaled = preds_train * label_std + label_mean
        test_labels_unscaled = self.labels_test * label_std + label_mean
        train_labels_unscaled = self.labels_train * label_std + label_mean
        test_length = len(test_labels_unscaled)

        std_train_labels = np.std(self.labels_train)

        # skip the last two values as they are anomalous
        self.mse_test = mean_squared_error(
            self.labels_test[self.n_steps + (self.time_delta - 1):-2:],
            self.preds[:len(self.labels_test[self.n_steps + (self.time_delta - 1):-2:])])  # / std_train_labels
        # TODO: fix indicies
        self.mse_train = mean_squared_error(
            self.labels_train[self.n_steps + (self.time_delta - 1):],
            preds_train[:len(self.labels_train[self.n_steps + (self.time_delta - 1):])])  # / std_train_labels

        self.mae_test = mean_absolute_error(
            test_labels_unscaled[self.n_steps + (self.time_delta - 1):-2],
            preds_unscaled[:len(test_labels_unscaled[self.n_steps + (self.time_delta - 1):-2])])

        self.mae_train = mean_absolute_error(
            train_labels_unscaled[self.n_steps + (self.time_delta - 1):],
            preds_train_unscaled[:len(self.labels_train[self.n_steps + (self.time_delta - 1):])])

        self.mae_test_right = mean_absolute_error(
            test_labels_unscaled[-int(test_length / 2):-2],
            preds_unscaled[-int(test_length / 2):-2])

        if (not self.model_saved) and (self.mae_test_right * 1.01 <= self.best_error):
            # save model checkpoint
            save_path_str = os.path.join(self.model_dir, '{n}_model_best.ckpt'.format(n=self.experiment_name))
            save_path = self.saver.save(self.sess, save_path_str)
            print("Model saved in file: %s" % '{n}_model_best.ckpt'.format(n=self.experiment_name))

        print('\n********')
        print('Train MSE (%std): {}'.format(self.mse_train))
        print('Test MSE (%std): {}'.format(self.mse_test))
        print('Train MAE [MGD]: {}'.format(self.mae_train))
        print('Test MAE [MGD]: {}'.format(self.mae_test))
        print('Test max error [MGD]: {}'.format(
            self.get_max_error(test_labels_unscaled[self.n_steps + (self.time_delta - 1):-2],
                               preds_unscaled[:len(test_labels_unscaled[self.n_steps + (self.time_delta - 1):-2])])))
        print('Test MAE (right) [MGD]: {}'.format(self.mae_test_right))
        print('********\n')

        plt.plot(self.times_train, self.labels_train * label_std + label_mean, 'b')
        plt.plot(self.times_test, self.labels_test * label_std + label_mean, 'b')
        # self.times_test.append(self.times_test[-1] + datetime.timedelta(days=1))
        # plt.plot(self.times_test[:self.n_steps], np.ones((self.n_steps)), "ro")

        plt.plot(self.times_test[self.n_steps + (self.time_delta - 1):],
                 self.preds[:len(self.times_test[self.n_steps
                                                 + (self.time_delta - 1):])] * label_std + label_mean, "r")
        if (PLATFORM.startswith('darwin')) & (not self.grid_mode):
            plt.show()
        plt.savefig(os.path.join(self.image_dir, "plot_" + str(num) + ".png"))
        plt.clf()

    def train(self, name='', train_num=0):
        print("training model")
        logging.info("training model")
        step = 1
        self.model_saved = False
        # Keep training until reach max iterations
        while step * self.batch_size < self.training_iters:
            try:
                batch_x, batch_y = self.get_batch()
                batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
                batch_y = batch_y.reshape((self.batch_size, self.n_classes * self.output_steps))

                # Run optimization op (backprop)
                _, s = self.sess.run([self.optimizer, self.merged_summary],
                                     feed_dict={self.x: batch_x, self.y: batch_y,
                                                self.keep_prob: self.keep_prob_param})
                if step % self.save_step == 0:
                    # save model checkpoint
                    save_path_str = os.path.join(self.model_dir,
                                                 '{n}_model{s}.ckpt'.format(n=self.experiment_name,
                                                                            s=step))
                    save_path = self.saver.save(self.sess, save_path_str)
                    print("Model saved in file: %s" % '{n}_model{s}.ckpt'.format(n=self.experiment_name,
                                                                                 s=step,
                                                                                 ))
                    log_save_path_str = os.path.join("./logs/", str(name), 'model')
                    self.saver.save(self.sess, log_save_path_str)

                    self.writer.add_summary(s, step + (train_num * self.training_iters / self.batch_size))
                    # self.writer.flush()
                    model_saved = True
                if step % self.display_step == 0:
                    # Calculate batch average error
                    acc = self.sess.run(self.average_error, feed_dict={self.x: batch_x,
                                                                       self.y: batch_y,
                                                                       self.keep_prob: self.keep_prob_param})
                    # Calculate batch loss
                    print("Iter " + str(step * self.batch_size) +
                          ", Average Error = " +
                          "{:.5f}".format(acc))
                step += 1
            except KeyboardInterrupt:
                continue

        print("Optimization Finished!")

    def restore(self, fn):
        # Restore variables from disk.
        self.saver = tf.train.import_meta_graph(fn + '.meta')
        self.saver.restore(self.sess, fn)
        print("Model restored.")

    def predict_flow(self, data):
        data_nump = np.array(data)
        data_nump = np.subtract(data_nump, self.mean)
        data_nump = np.divide(data_nump, self.std)
        predictions = []
        for _ in range(7):
            data_nump = np.reshape(data_nump, (1, self.n_steps, self.n_input))
            self.preds = self.sess.run(self.pred, feed_dict={self.x: data_nump, self.keep_prob: 1.})
            data_nump = np.roll(data_nump, -1)
            data_nump[-1] = self.preds[0]
            predictions.append(self.preds[0])

        predictions = np.array(predictions)
        predictions = np.multiply(predictions, self.std)
        predictions = np.add(predictions, self.mean)
        return predictions


    def get_max_error(self, y_vals, y_preds):
        max_error = 0
        for i, j in zip(y_vals, y_preds):
            error = abs(i - j)
            if error > max_error:
                max_error = error
        return max_error

    def generate_guitar(self, n_samples = 44100):
        seed, _ = self.get_batch()

        samples = []
        for i in seed:
            samples.append(i)

        seed = np.array(seed[-1]).reshape(1, self.n_steps, 1)
        for i in range(n_samples):
            seed = np.array(seed).reshape(1, self.n_steps, 1)
            time_before = time.time()
            next_sample = self.sess.run(self.pred, feed_dict={self.x: seed, self.keep_prob: 1.})
            time_after = time.time()
            print "sample: {} out of: {}, run time: {}".format(i, n_samples, (time_after - time_before))
            seed = np.append(seed, [[next_sample]])
            seed = np.delete(seed, 0, 0)
            samples.append(next_sample)
        samples = np.array(samples, dtype=np.int16)
        scipy.io.wavfile.write("out.wav", self.sample_rate, samples)




def run_grid(input_params, grid_name, grid_params):
    # Define base model configuration
    base_model_config_dict = dict()
    base_model_config_dict['input_params'] = input_params

    # Initialize dictionaries
    param_dict = dict()
    index_dict = dict()
    results_dict = dict()

    # Create a dictionary of lists of parameter ranges
    for i, p in enumerate(grid_params):
        if p == 'cell_type':
            print('CELL TYPE! {}'.format(grid_params[p]))
            param_dict[p] = np.arange(0, len(grid_params[p]['value_list']), 1)
        else:
            p_min = grid_params[p]['min']
            p_max = grid_params[p]['max']
            p_step = grid_params[p]['step']

            if (isinstance(p_min, int)) and (isinstance(p_max, int)) and (isinstance(p_step, int)):
                p_array = np.array(range(p_min, p_max + p_step, p_step))
            else:
                p_array = np.arange(p_min, p_max + p_step, p_step)

            param_dict[p] = p_array

    for i, p in enumerate(param_dict):
        index_dict[i] = p
        i += 1

    # Create dense meshgrid
    n_params = len(param_dict)
    mesh_params = np.array(np.meshgrid(*(v for k, v in param_dict.iteritems()))).T.reshape(-1, n_params)
    n_mesh = len(mesh_params)

    # Create a dictionary for each experiment and run new model
    for param_i, r in enumerate(mesh_params):
        np.random.seed(42)
        model_config_dict = dict(base_model_config_dict)
        experiment_name = 'G_{}'.format(grid_name)
        for i, v in enumerate(r):
            param_name = index_dict[i]
            if param_name == 'cell_type':
                model_config_dict['input_params'][param_name] = grid_params[param_name]['value_list'][int(v)]
                print('CELL TYPE! {}'.format(grid_params[param_name]['value_list'][int(v)]))
            else:
                model_config_dict['input_params'][param_name] = v

            v_str = str(v).replace('.', 'o')
            experiment_name = experiment_name + '_{p}{v}'.format(p=param_name,
                                                                 v=v_str)
        model_config_dict['experiment_name'] = experiment_name

        print('\n* Running new experiment: {}'.format(experiment_name))
        print('  {}/{}'.format(param_i, n_mesh))
        print(model_config_dict)

        lstm_forecast = LSTMForecasting()
        lstm_forecast.load_config(model_config_dict)
        lstm_forecast.load_features(os.path.join("..", ".."))
        lstm_forecast.tf_graph(name=experiment_name)
        lstm_forecast.grid_mode = True

        for j in range(input_params['training_cycles']):
            print("{}: Training cycle {}".format(experiment_name, j))
            total_iter = (j + 1) * lstm_forecast.training_iters

            lstm_forecast.train(train_num=j, name=experiment_name)
            # Some models can become unstable, resulting in nans or infs.
            try:
                lstm_forecast.test(j)
            except:
                break

            mse_test = lstm_forecast.mse_test
            mse_train = lstm_forecast.mse_train
            mae_test = lstm_forecast.mae_test
            mae_train = lstm_forecast.mae_train
            mae_test_right = lstm_forecast.mae_test_right

            model_result = dict(model_config_dict)
            model_result['total_iterations'] = total_iter
            model_result['mse_test'] = mse_test
            model_result['mse_train'] = mse_train
            model_result['mae_test'] = mae_test
            model_result['mae_train'] = mae_train
            model_result['mae_test_right'] = mae_test_right

            # model_result['prediction_time'] = lstm_forecast.prediction_time

            results_dict[experiment_name] = model_result

        with open(os.path.join('grid_results', '{}.json'.format(grid_name)), 'w') as fp:
            json.dump(results_dict, fp)

        tf.reset_default_graph()
        del (lstm_forecast)


def load_grid(grid_config_name):
    if grid_config_name[-5:] == '.json'.lower():
        grid_config_name = grid_config_name[:-5]

    # Open config file
    with open('{}.json'.format(grid_config_name)) as json_str:
        json_grid_config = json.load(json_str)

    input_params = json_grid_config['input_params']
    grid_name = json_grid_config['grid_name']
    grid_params = json_grid_config['grid_params']
    print('grid_config')
    print(json_grid_config)

    i_name = 0
    result_file = os.path.join("grid_results", '{}'.format(grid_name))
    while os.path.exists(result_file):
        i_name += 1
        result_file = os.path.join("grid_results", '{n}_{i}'.format(n=grid_name, i=i_name))

    grid_name = '{n}_{i}'.format(n=grid_name, i=i_name)

    return input_params, grid_name, grid_params


if __name__ == '__main__':
    lstm_forecast = LSTMForecasting()
    lstm_forecast.load_features()
    lstm_forecast.tf_graph()
    lstm_forecast.restore("./models/default/default_model8320.ckpt")
    # lstm_forecast.train()
    lstm_forecast.generate_guitar()


