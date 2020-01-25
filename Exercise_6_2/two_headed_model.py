import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def generate_sine_dataset(seed):
    np.random.seed(seed)

    X = np.random.uniform(-np.pi, np.pi, 100)
    Y = []
    for x in X:
        if x < 0:
            eta = 0.2
        else:
            eta = 0.5
        Y.append(np.sin(0.5 * x) + np.random.normal(0, eta))

    X_train = np.array(X)
    Y_train = np.array(Y)
    '''X_test = np.array(X[90:100])
    Y_test = np.array(Y[90:100])'''

    X_test = np.linspace(-np.pi, np.pi, 50)[:, np.newaxis]
    Y_test = np.sin(0.5 * X_test)

    return X_train.reshape(100, 1), Y_train.reshape(100, 1), X_test.reshape(50, 1), Y_test.reshape(50, 1)


def gaussian_log_likelihood(Y, mu, sigma=0.01):
    return 0.5 / 100 * tf.reduce_sum(tf.log(sigma) + (mu - Y) ** 2 / sigma)


def neural_net(x):
    # hidden layer 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)  # activation
    # hideen layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)  # activation
    # output layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    mu, sigma = tf.nn.moments(out_layer, [0])

    #sigma = tf.nn.relu(sigma)

    return (out_layer), mu, sigma


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = generate_sine_dataset(17)

    X = tf.placeholder("float", [None, 1])
    Y = tf.placeholder("float", [None, 1])

    weights = {
        'h1': tf.Variable(tf.random_normal([1, 20])),  # 4 inputs 10  nodes in h1 layer
        'h2': tf.Variable(tf.random_normal([20, 10])),  # 10 nodes in h2 layer
        'out': tf.Variable(tf.random_normal([10, 1])),  # 1 ouput label
        'mu_sigma': tf.Variable(tf.random_normal([1, 2]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([20])),
        'b2': tf.Variable(tf.random_normal([10])),
        'out': tf.Variable(tf.random_normal([1])),
        'mu_sigma': tf.Variable(tf.random_normal([2]))
    }

    Y_hat, mu, sigma = neural_net(X)
    loss_op = 0.5 / 100 * tf.reduce_sum(tf.log(sigma) + (mu - y_train) ** 2 / sigma)  # loss function
    #loss_op = tf.keras.losses.MSE(y_train,Y_hat)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)  # define optimizer # play around with learning rate
    train_op = optimizer.minimize(loss_op)  # minimize loss
    init = tf.global_variables_initializer()
    epoch = 1000

    with tf.Session() as sess:
        sess.run(init)
        for i in range(0, epoch):
            sess.run(train_op, feed_dict={X: x_train, Y: y_train})
            loss = sess.run(loss_op, feed_dict={X: x_train, Y: y_train})
            if (i % 100 == 0):
                print("epoch no" + str(i), (loss))
            pred = sess.run(tf.convert_to_tensor(Y_hat), feed_dict={X: x_test})

    # TODO: expected function?, correct interval, scatter
    #       plot mean, variance

    '''plt.plot((pred), color='red', label='Prediction')
    plt.plot(y_test, color='blue', label='Orignal')
    plt.legend(loc='upper left')
    plt.show()'''

    plt.figure()
    plt.title("Prediction")
    plt.scatter(x_train.ravel(), y_train.ravel(), label="Training set (noisy)")
    plt.plot(x_test.ravel(), y_test.ravel(), lw=3, label="True function")
    plt.plot(x_test.ravel(), pred.ravel(), lw=3,
             label="Prediction")
    plt.legend(loc="best")
    plt.show()
