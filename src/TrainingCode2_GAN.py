import numpy as np
from numpy import expand_dims,zeros,ones,vstack
from numpy.random import randn, randint
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU,Dropout
from matplotlib import pyplot


def build_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def build_generator(latent_dim):
	model = Sequential()
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model


def build_gan(gen_model, disc_model):
	disc_model.trainable = False
	model = Sequential()
	model.add(gen_model)
	model.add(disc_model)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def generate_real_samples(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, 1))
	return X, y


def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


def generate_fake_samples(generator_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	X = generator_model.predict(x_input)
	y = zeros((n_samples, 1))
	return X, y
    

def train(generator_model, discriminator_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    history = list()
    batch_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        tmp_hist = list()
        for j in range(batch_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(generator_model, latent_dim, half_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            d_loss, _ = discriminator_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            tmp_hist.append(g_loss)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_epoch, d_loss, g_loss))
            
        history.append(np.mean(tmp_hist))
        if (i+1) % 10 == 0:
            filename = 'generator_model_%03d.h5' % (i + 1)
            generator_model.save(filename)
    return history

latent_dim = 10
discriminator_model = build_discriminator()
generator_model = build_generator(latent_dim)
gan_model = build_gan(generator_model, discriminator_model)

(trainX, _), (_, _) = mnist.load_data()
dataset = expand_dims(trainX, axis=-1)
dataset = dataset.astype('float32')
dataset = dataset / 255.0

history = train(generator_model, discriminator_model, gan_model, dataset, latent_dim, n_epochs= 100)

pyplot.plot(history, label='GAN loss')
pyplot.legend()
pyplot.savefig('gan_loss.png')
pyplot.close()