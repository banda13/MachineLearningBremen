import os
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
img_dir = 'images/'

if not os.path.isdir(img_dir):
    os.mkdir(img_dir)

class VAE(object):

    def __init__(self, latent_dimension, img_shape):
        self.latent_dim = latent_dimension
        self.img_shape = img_shape
        self.origin_dim = self.img_shape * self.img_shape

        # placeholders
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.inputs = None
        self.outputs = None
        self.history = None

        # network parameters
        # number of nodes in the hidden layers
        self.intermediate_dim = 512
        # size of the filters in the conv layers
        self.filters = 64
        # kernel size in conv layers
        self.kernel_size = 3

    def compile_mlp_model(self):
        print('Compiling MLP network')

        # creating encoder
        self.inputs = tf.keras.layers.Input(shape=(self.origin_dim,), name='encoder_input')

        # hidden fully connected layers with small dropout between the layers
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(self.inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x)

        # parallel layers: mean and variance
        z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(x)

        # applying reparametrization trick
        z = tf.keras.layers.Lambda(self.reparametrization, output_shape=(self.latent_dim,), name='z')(
            [z_mean, z_log_var])
        self.encoder = tf.keras.models.Model(self.inputs, [z_mean, z_log_var, z], name='encoder')

        # plot the architecture of the encoder
        tf.keras.utils.plot_model(self.encoder, to_file=img_dir + 'vae_mlp_encoder.png', show_shapes=True)

        # creating decoder
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name='z_sampling')

        # decoder hidden layers, similar to the encoder
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x) # extra
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x) # extra

        self.outputs = tf.keras.layers.Dense(self.origin_dim, activation='sigmoid')(x)
        self.decoder = tf.keras.models.Model(latent_inputs, self.outputs, name='decoder')
        tf.keras.utils.plot_model(self.decoder, to_file=img_dir + 'vae_mlp_decoder.png', show_shapes=True)

        # creating vae
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = tf.keras.models.Model(self.inputs, self.outputs, name='vae_mlp')

        # loss function
        reconstruction_loss = tf.keras.losses.mse(self.inputs, self.outputs)
        reconstruction_loss *= self.origin_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        tf.keras.utils.plot_model(self.vae, to_file=img_dir + 'vae.png', show_shapes=True)

    def compile_cnn_model(self):
        print('Compiling CNN network')

        # create encoder
        self.inputs = tf.keras.layers.Input(shape=(self.img_shape, self.img_shape, 1))
        x = self.inputs
        # adding convolutional layers on the the of the encoder
        for i in range(2):
            self.filters *= 2
            x = tf.keras.layers.Conv2D(filters=self.filters,
                                       kernel_size=self.kernel_size,
                                       activation='relu',
                                       strides=2,
                                       padding='same')(x)

        shape = K.int_shape(x)
        # flattening the convolutional layers and pass it to the fully connected layers
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x)

        z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(x)

        z = tf.keras.layers.Lambda(self.reparametrization, output_shape=(self.latent_dim,), name='z')(
            [z_mean, z_log_var])
        self.encoder = tf.keras.models.Model(self.inputs, [z_mean, z_log_var, z], name='encoder')
        tf.keras.utils.plot_model(self.encoder, to_file=img_dir + 'vae_cnn_encoder.png', show_shapes=True)

        # instantiate decoder
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(x)

        x = tf.keras.layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)

        # reshape a data the the proper dimension, and applying deconvolution
        x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)
        for i in range(2):
            x = tf.keras.layers.Conv2DTranspose(filters=self.filters,
                                                kernel_size=self.kernel_size,
                                                activation='relu',
                                                strides=2,
                                                padding='same')(x)
            self.filters //= 2

        self.outputs = tf.keras.layers.Conv2DTranspose(filters=1,
                                                       kernel_size=self.kernel_size,
                                                       activation='sigmoid',
                                                       padding='same',
                                                       name='decoder_output')(x)

        # create decoder model
        self.decoder = tf.keras.models.Model(latent_inputs, self.outputs, name='decoder')
        tf.keras.utils.plot_model(self.decoder, to_file=img_dir + 'vae_cnn_decoder.png', show_shapes=True)

        # create vae
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = tf.keras.models.Model(self.inputs, self.outputs, name='vae')

        # loss
        reconstruction_loss = tf.keras.losses.mse(K.flatten(self.inputs), K.flatten(self.outputs))
        reconstruction_loss *= image_size * image_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='rmsprop', metrics=['accuracy'])
        tf.keras.utils.plot_model(self.vae, to_file=img_dir + 'vae_cnn.png', show_shapes=True)

    def reparametrization(self, args):
        # sample the distribution of the latent dimension
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def train(self, x_train, epochs, batch_size, x_test):
        # use early stopping to prevent overfitting by interrupting the training process
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        self.history = self.vae.fit(x_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(x_test, None),
                                    callbacks=[es])
        return self.history

    def plot_training(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(img_dir + 'traning_loss.png')
        plt.show()
        plt.close()

    def plot_latent_space(self, test_images):
        z_mean, _, _ = self.encoder.predict(test_images,
                                            batch_size=128)
        x = z_mean[:, 0]
        y = z_mean[:, 1]
        unique = np.unique(test_labels)
        colors = [plt.cm.jet(i / float(len(unique) - 1)) for i in range(len(unique))]
        for i, u in enumerate(unique):
            xi = [x[j] for j in range(len(x)) if test_labels[j] == u]
            yi = [y[j] for j in range(len(y)) if test_labels[j] == u]
            plt.scatter(xi, yi, c=colors[i], label=str(labels[u]))
        plt.legend()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(img_dir + "latent_space.png")
        plt.show()
        plt.close()

    def plot_data_space(self):
        n = 30
        figure = np.zeros((self.img_shape * n, self.img_shape * n))
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(self.img_shape, self.img_shape)
                figure[i * self.img_shape: (i + 1) * self.img_shape,
                j * self.img_shape: (j + 1) * self.img_shape] = digit

        plt.figure(figsize=(10, 10))
        start_range = self.img_shape // 2
        end_range = (n - 1) * self.img_shape + start_range + 1
        pixel_range = np.arange(start_range, end_range, self.img_shape)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(img_dir + 'data_space.png')
        plt.show()
        plt.close()


if __name__ == '__main__':

    MODE = 'cnn'  # cnn or mlp for now
    if MODE not in ['cnn', 'mlp']:
        raise Exception('Unrecognized mode. valid values: mlp or cnn')

    # loading images
    (train_images, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    print('Train size %d, test size %d' % (len(train_images), len(test_images)))

    # preprocessing
    if MODE == 'mlp':
        image_size = train_images.shape[1]
        original_dim = image_size * image_size
        train_images = np.reshape(train_images, [-1, original_dim])
        test_images = np.reshape(test_images, [-1, original_dim])
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
    elif MODE == 'cnn':
        image_size = train_images.shape[1]
        train_images = np.reshape(train_images, [-1, image_size, image_size, 1])
        test_images = np.reshape(test_images, [-1, image_size, image_size, 1])
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255

    EPOCHS = 50
    BATCH_SIZE = 64
    LATENT_SPACE = 2

    # training
    vae = VAE(LATENT_SPACE, image_size)
    if MODE == 'mlp':
        vae.compile_mlp_model()
    elif MODE == 'cnn':
        vae.compile_cnn_model()
    vae.train(train_images, EPOCHS, BATCH_SIZE, test_images)
    vae.plot_training()

    # testing & visualizing the results
    vae.plot_latent_space(test_images)
    vae.plot_data_space()
