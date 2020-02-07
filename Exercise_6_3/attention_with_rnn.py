import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# inherit from Keras models as our attention we are using a small nn
class Attention(tf.keras.Model):

    def __init__(self, units):
        super(Attention, self).__init__()
        # dense layer + 1 output layer with one output unit
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # calculate the weights using softmax activation
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # all encoded states of the rnn are equally valuable
        # so we are using the weighted sum of these encoded states
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


DATA_FODLER = 'FiveCitiePMData/'
IMAGE_FOLDER = "images/"

if not os.path.isdir(IMAGE_FOLDER):
    os.mkdir(IMAGE_FOLDER)

# needed for backwards compatibility
tf.compat.v1.enable_eager_execution()


class RNN(object):

    def __init__(self, sliding_window_size, rnn, attention = True):
        self.trainX = None
        self.testX = None
        self.trainY = None
        self.testY = None

        self.sliding_window_size = sliding_window_size

        self.history = None
        self.model = None

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = None

        self.rnn = rnn
        self.with_attention = attention

    def load_and_preprocess(self):

        # parse csv-s into pd dataframe
        data_source_names = os.listdir(DATA_FODLER)
        data_sources = [pd.read_csv(DATA_FODLER + f, delimiter=',') for f in data_source_names]

        # filter out the columns
        pm_dfs = [d.filter(like='PM_') for d in data_sources]
        pm_dfs = [d.mean(axis=1) for d in pm_dfs]
        date_dfs = [d.filter(items=['year', 'month', 'day', 'hour']) for d in data_sources]
        dfs = [pd.concat([d1, d2], axis=1) for d1, d2 in zip(pm_dfs, date_dfs)]

        df = dfs[0]
        for d in dfs[1:]:
            df = pd.merge(df, d, on=['year', 'month', 'day', 'hour'])

        df = df.drop(['year', 'month', 'day', 'hour'], axis=1)
        df = df.mean(axis=1)

        # drop nan rows
        df = df.dropna()

        # convert it numpy array
        self.dataset = df.values.reshape(-1, 1)

        # normalize the data
        self.dataset = self.scaler.fit_transform(self.dataset)

        # split into train and test
        train_factor = 0.8
        train_size = int(len(self.dataset) * train_factor)
        train, test = self.dataset[0:train_size, :], self.dataset[train_size:len(self.dataset), :]

        # format into time series
        self.trainX, self.trainY = self.create_dataset(train, self.sliding_window_size)
        self.testX, self.testY = self.create_dataset(test, self.sliding_window_size)

        # reshape input to be [samples, time steps, features]
        self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], self.trainX.shape[1], 1))
        self.testX = np.reshape(self.testX, (self.testX.shape[0], self.testX.shape[1], 1))

    def create_dataset(self, dataset, sliding_window_size):
        dataX, dataY = [], []
        for i in range(len(dataset) - sliding_window_size - 1):
            a = dataset[i:(i + sliding_window_size), 0]
            dataX.append(a)
            dataY.append(dataset[i + sliding_window_size, 0])
        return np.array(dataX), np.array(dataY)

    def compile_model(self):
        input = tf.keras.layers.Input(shape=(self.sliding_window_size, 1))
        return_seq = self.with_attention

        attention = Attention(16)
        if self.rnn == 'lstm':
            lstm, forward_h, forward_c = tf.keras.layers.LSTM(64, return_sequences=return_seq, return_state=True)(input)
        elif self.rnn == 'rnn':
            lstm, forward_h = tf.keras.layers.SimpleRNN(64, return_sequences=return_seq, return_state=True)(input)
        elif self.rnn == 'gru':
            lstm, forward_h = tf.keras.layers.GRU(64, return_sequences=return_seq, return_state=True)(input)
        else:
            raise Exception('Invalid recurrent layer: ', self.rnn)

        # state concatenations are needed for biderectional lstm
        # state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        # state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

        if self.with_attention:
            context_vector, attention_weights = attention(lstm, forward_h)
        else:
            context_vector = lstm
        hidden = tf.keras.layers.Dense(32)(context_vector)
        hidden = tf.keras.layers.Dense(32)(hidden)
        output = tf.keras.layers.Dense(1)(hidden)

        self.model = tf.keras.models.Model(inputs=input, outputs=output)
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=0.00005))
        print('model compiled without error')

    def train(self, epochs=10, batch_size=1, verbose=1):
        self.history = self.model.fit(self.trainX, self.trainY,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(self.testX, self.testY),
                                      # callbacks=[es],
                                      verbose=verbose)

    def evaluate(self):
        # make predictions
        self.train_predict = self.model.predict(self.trainX)
        self.test_predict = self.model.predict(self.testX)

        # invert predictions
        self.train_predict = self.scaler.inverse_transform(self.train_predict)
        self.trainY = self.scaler.inverse_transform([self.trainY])
        self.test_predict = self.scaler.inverse_transform(self.test_predict)
        self.testY = self.scaler.inverse_transform([self.testY])

        # calculate root mean squared error
        train_score = math.sqrt(tf.keras.losses.mean_squared_error(self.trainY[0], self.train_predict[:, 0]))
        print('Train Score: %.4f RMSE' % train_score)
        test_score = math.sqrt(tf.keras.losses.mean_squared_error(self.testY[0], self.test_predict[:, 0]))
        print('Test Score: %.4f RMSE' % test_score)
        return train_score, test_score

    def plot(self):
        # plot training loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(IMAGE_FOLDER + 'training_loss.png')
        plt.show()
        plt.close()

        # shift train predictions for plotting
        train_predict_plot = np.empty_like(self.dataset)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[self.sliding_window_size:len(self.train_predict) + self.sliding_window_size, :] = self.train_predict

        # shift test predictions for plotting
        test_predict_plot = np.empty_like(self.dataset)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(self.train_predict) + (self.sliding_window_size * 2) + 1:len(self.dataset) - 1, :] = self.test_predict

        # plot baseline and predictions
        plt.plot(self.scaler.inverse_transform(self.dataset), label='origin')
        plt.plot(train_predict_plot, label='train')
        plt.plot(test_predict_plot, label='test')
        plt.title('Origin data vs predictions')
        plt.legend()
        plt.show()
        plt.savefig(IMAGE_FOLDER + 'baseline_and_prediction.png')
        plt.close()


if __name__ == '__main__':

    # choose parameters
    recurrent_layer = 'gru'  # lstm or rnn or gru
    attention = True  # attention is enabled

    rnn = RNN(sliding_window_size=5, rnn=recurrent_layer, attention=attention)
    rnn.load_and_preprocess()
    rnn.compile_model()
    rnn.train(epochs=50, batch_size=64)
    train_loss, test_loss = rnn.evaluate()
    rnn.plot()