import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

        # other fixed params, but feel free to change it
        self.intermediate_dim = 512

    def compile_model(self):
        # creating encoder
        self.inputs = tf.keras.layers.Input(shape=(self.origin_dim,), name='encoder_input')
        # self.inputs = tf.keras.layers.Input(shape=(self.img_shape, self.img_shape, 1))
        # conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(self.inputs)
        # flatten = tf.keras.layers.Flatten()(conv1)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(self.inputs)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = tf.keras.layers.Lambda(self.reparametrization, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        self.encoder = tf.keras.models.Model(self.inputs, [z_mean, z_log_var, z], name='encoder')
        tf.keras.utils.plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # creating decoder
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        """x = tf.keras.layers.Reshape(target_shape=(14, 14, 32))(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding="SAME",
            activation='relu')(x)
        self.outputs = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=(1, 1),
            padding="SAME")(x)
        """
        self.outputs = tf.keras.layers.Dense(self.origin_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = tf.keras.models.Model(latent_inputs, self.outputs, name='decoder')
        tf.keras.utils.plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = tf.keras.models.Model(self.inputs, self.outputs, name='vae_mlp')
        tf.keras.utils.plot_model(self.vae, to_file='vae.png', show_shapes=True)


        # loss
        reconstruction_loss = tf.keras.losses.mse(self.inputs, self.outputs)
        reconstruction_loss *= self.origin_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam', metrics=['accuracy'])

    def reparametrization(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def train(self, x_train, epochs, batch_size, x_test):
        self.history = self.vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        return self.history

    def plot_training(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('traning_loss.png')
        plt.show()

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
        plt.savefig("latent_space.png")
        plt.show()

    def plot_data_space(self):
        n = 30
        figure = np.zeros((self.img_shape * n, self.img_shape * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
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
        plt.savefig('data_space.png')
        plt.show()

if __name__ == '__main__':
    # loading images
    (train_images, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    print('Train size %d, test size %d' % (len(train_images), len(test_images)))

    """
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    EPOCHS = 3
    TEST_BUF = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)
    """

    # preprocessing
    image_size = train_images.shape[1]
    original_dim = image_size * image_size
    train_images = np.reshape(train_images, [-1, original_dim])
    test_images = np.reshape(test_images, [-1, original_dim])
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # training
    vae = VAE(2, image_size)
    vae.compile_model()
    vae.train(train_images, 50, 128, test_images)
    vae.plot_training()

    # testing and visualizing the results
    vae.plot_latent_space(test_images)
    vae.plot_data_space()