import keras
from keras.layers import Input, Reshape, LSTM, Dense, Lambda, Dropout, Conv2DTranspose
from keras.models import Model

from keras.datasets import mnist
from keras import backend as K
import numpy as np

import matplotlib.pyplot as plt

original_dim = (28,28,1)
latent_dim = 10
num_channels = 1
epochs = 50

# ENCODER
inputs = Input(shape=original_dim)
x = Reshape((28, 28))(inputs)
h = LSTM(units = 256, activation = 'relu',return_sequences=False)(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mean, z_log_sigma])
encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

#DECODER
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

x = Dense(16, activation='relu')(decoder_input)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Reshape((4, 4, 8))(x)
x = Conv2DTranspose(filters = 8, kernel_size= 2, padding = 'valid', activation='relu',strides=(2, 2))(x)
x = Conv2DTranspose(filters = 4, kernel_size= 5, padding = 'valid', activation='relu',strides=(1, 1))(x)
x = Conv2DTranspose(filters = 2, kernel_size= 2, padding = 'valid', activation='relu',strides=(2, 2))(x)
x = Conv2DTranspose(filters = 1, kernel_size= 5, padding = 'valid', activation='relu',strides=(1, 1))(x)

decoder = Model(decoder_input, x, name='decoder')

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = keras.losses.binary_crossentropy(K.flatten(inputs),K.flatten(outputs))
reconstruction_loss *= (original_dim[0] * original_dim[1])

kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

#vae_loss = K.mean(reconstruction_loss + kl_loss)
vae_loss = K.mean(reconstruction_loss) + 2 * K.mean(kl_loss)
vae.add_loss(vae_loss)
vae.add_metric(kl_loss, name="kl_param")
vae.compile(optimizer='adam')

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# filepath="/content/sample_data/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5" #File name includes epoch and validation accuracy.

# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stop = EarlyStopping(monitor='val_loss', min_delta = 0.5, patience=4, verbose=1, mode = "min", restore_best_weights = True)

#log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

callbacks_list = [early_stop]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

history = vae.fit(x_train, x_train, epochs=epochs, batch_size=32, shuffle=True, validation_data=(x_test, x_test), callbacks=callbacks_list)

encoder.save('encoder.h5') 



