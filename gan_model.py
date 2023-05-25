import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

class GAN:
    def __init__(self, latent_dim=100, img_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
        self.combined = self._build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    def _build_generator(self):
        model = Sequential()
        model.add(layers.Dense(256 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(layers.Reshape((7, 7, 256)))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation("relu"))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(self.img_shape[2], kernel_size=3, padding="same", activation="tanh"))
        return model

    def _build_discriminator(self):
        model = Sequential()
        model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def _build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def train(self, X_train, epochs, batch_size=128, sample_interval=50):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            if epoch % sample_interval == 0:
                print (f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.2f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.2f}]")
                self._sample_images(epoch)

    def _sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale images 0 - 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"images/mnist_{epoch}.png")
        plt.close()

if __name__ == "__main__":
    # Load MNIST dataset
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    # Create a directory for generated images
    os.makedirs("images", exist_ok=True)

    gan = GAN()
    gan.train(X_train, epochs=4000, batch_size=32, sample_interval=200)
