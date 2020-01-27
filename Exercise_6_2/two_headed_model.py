import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def plot_mean_and_CI(x, mean, variance, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    ub = mean + variance
    lb = mean - variance
    plt.fill_between(x, ub, lb,
                     color=color_shading, alpha=.5)
    plt.plot(x_test.ravel(), mean.ravel(), color_mean, lw=3,
             label="Confidence Interval: "+ str(x.shape[0])+" samples")


def generate_sine_dataset(seed, n_sample=100):
    np.random.seed(seed)

    train_shape = int(3*n_sample/4)
    test_shape = int(n_sample / 4)

    X = np.random.uniform(-np.pi, np.pi, train_shape)
    Y = []
    for x in X:
        if x < 0:
            eta = 0.2
        else:
            eta = 0.5
        Y.append(np.sin(0.5 * x) + np.random.normal(0, eta))

    X_train = np.array(X)
    Y_train = np.array(Y)

    X_test = np.random.uniform(-np.pi, np.pi, test_shape)
    X_test.sort()
    Y_test = np.sin(0.5 * X_test)

    return X_train.reshape(train_shape, 1), Y_train.reshape(train_shape, 1), \
           X_test.reshape(test_shape, 1), Y_test.reshape(test_shape, 1)


def gaussian_log_likelihood(Y, mu, sigma):
    sum = tf.log(sigma) + (mu - Y) ** 2 / sigma
    return 0.5 / Y.shape[0] * tf.reduce_sum(sum)



class two_headed_model(object):
    def __init__(self):
        self.X = tf.placeholder("float", [None, 1])
        self.MU = tf.placeholder("float", [None, 1])
        self.SIGMA = tf.placeholder("float", [None, 1])

    def construct_model(self):
        weights = {
            'h1': tf.Variable(tf.random_normal([1, 50], stddev=0.01)),
            'h2': tf.Variable(tf.random_normal([50, 20], stddev=0.01)),
            'mu1': tf.Variable(tf.random_normal([20, 60], stddev=0.001)),
            #'mu2': tf.Variable(tf.random_normal([60, 30], stddev=0.001)),
            'mu_out': tf.Variable(tf.random_normal([60, 1], stddev=0.001)),
            'sigma1': tf.Variable(tf.random_normal([20, 40], stddev=0.001)),
            #'sigma2': tf.Variable(tf.random_normal([40, 20], stddev=0.001)),
            'sigma_out': tf.Variable(tf.random_normal([40, 1], stddev=0.001))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([50], stddev=0.01)),
            'b2': tf.Variable(tf.random_normal([20], stddev=0.01)),
            'mu1': tf.Variable(tf.random_normal([60], stddev=0.001)),
            #'mu2': tf.Variable(tf.random_normal([30], stddev=0.001)),
            'mu_out': tf.Variable(tf.random_normal([1], stddev=0.001)),
            'sigma1': tf.Variable(tf.random_normal([40], stddev=0.001)),
            #'sigma2': tf.Variable(tf.random_normal([20], stddev=0.001)),
            'sigma_out': tf.Variable(tf.random_normal([1], stddev=0.001))
        }
        layer_1 = tf.add(tf.matmul(self.X, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)  # activation
        # hideen layer 2
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)  # activation

        mu_h1 = tf.add(tf.matmul(layer_2, weights['mu1']), biases['mu1'])
        mu_h1 = tf.nn.relu(mu_h1)

        '''mu_h2 = tf.add(tf.matmul(mu_h1, weights['mu2']), biases['mu2'])
        mu_h2= tf.nn.relu(mu_h2)'''

        mu_out = tf.add(tf.matmul(mu_h1, weights['mu_out']), biases['mu_out'])
        self.mu = mu_out

        sigma_h1 = tf.add(tf.matmul(layer_2, weights['sigma1']), biases['sigma1'])
        sigma_h1 = tf.nn.sigmoid(sigma_h1)

        '''sigma_h2 = tf.add(tf.matmul(sigma_h1, weights['sigma2']), biases['sigma2'])
        sigma_h2 = tf.nn.sigmoid(sigma_h2)'''

        sigma_out = tf.add(tf.matmul(sigma_h1, weights['sigma_out']), biases['sigma_out'])

        self.sigma = tf.nn.sigmoid(sigma_out)

    def compile(self, y_train, learning_rate=0.001):
        self.loss_op = gaussian_log_likelihood(y_train, self.mu, self.sigma)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

    def train(self, x_train, y_train, x_test, epoch=5000):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(0, epoch):
                sess.run(self.train_op,
                         feed_dict={self.X: x_train, self.MU: np.mean(y_train).reshape(1, 1),
                                    self.SIGMA: (np.std(y_train)**2).reshape(1, 1)})
                loss = sess.run(self.loss_op, feed_dict={self.X: x_train, self.MU: np.mean(y_train).reshape(1, 1),
                                                    self.SIGMA: np.std(y_train).reshape(1, 1)})
                if (i % 100 == 0):
                    print("epoch no " + str(i), (loss))
            pred = sess.run([self.mu, self.sigma], feed_dict={self.X: x_test})
            self.pred_mean = pred[0]
            self.pred_var = pred[1]

    def plot(self, x_train, y_train, x_test, y_test):
        plt.figure()
        plt.title("Prediction")
        plt.scatter(x_train.ravel(), y_train.ravel(), label="Training set (noisy): "
                                                            +str(x_train.shape[0])+" samples", s=10)
        plt.plot(x_test.ravel(), y_test.ravel(), 'g', lw=3, label="True function")
        plot_mean_and_CI(x_test.ravel(), self.pred_mean.ravel(), self.pred_var.ravel(), color_mean='y', color_shading='y')
        plt.legend(loc="best")
        plt.show()

if __name__ == "__main__":
    # point a)
    x_train, y_train, x_test, y_test = generate_sine_dataset(17, n_sample=100)

    two_headed = two_headed_model()
    two_headed.construct_model()
    two_headed.compile(y_train=y_train, learning_rate=0.006)
    two_headed.train(x_train=x_train, y_train=y_train, x_test=x_test, epoch=3000)
    two_headed.plot(x_train, y_train, x_test, y_test)
    
    # point b)
    i = 10
    while i < 1300:
        x_train, y_train, x_test, y_test = generate_sine_dataset(randomseed=23+i, n_sample=i)
        two_headed = two_headed_model()
        two_headed.construct_model()
        two_headed.compile(y_train=y_train, learning_rate=0.006)
        two_headed.train(x_train=x_train, y_train=y_train, x_test=x_test, epoch=3000)
        two_headed.plot(x_train, y_train, x_test, y_test)
        i *= 2
    
    #point c)

    shape = x_test.shape[0]
    x_test = np.random.uniform(-4*np.pi, 4*np.pi, shape)
    x_test.sort()
    x_test = x_test.reshape(shape, 1)
    y_test = np.sin(0.5 * x_test)

    two_headed = two_headed_model()
    two_headed.construct_model()
    two_headed.compile(y_train=y_train, learning_rate=0.006)
    two_headed.train(x_train=x_train, y_train=y_train, x_test=x_test, epoch=1000)
    two_headed.plot(x_train, y_train, x_test, y_test)