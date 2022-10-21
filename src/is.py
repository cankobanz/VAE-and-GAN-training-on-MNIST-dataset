# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.models import load_model
from skimage.transform import resize
from numpy import asarray
import numpy as np

decoder = load_model('generator_model_100.h5')

latent_dim = 10
n_samples = 500
np.random.seed(102)

rand_vec = np.random.randn(latent_dim * n_samples)
rand_vec = rand_vec.reshape(n_samples, latent_dim)
X = decoder.predict(rand_vec)
X = np.reshape(X, (n_samples,28,28))


def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		new_image = resize(image, new_shape, 0)
		images_list.append(new_image)
	return asarray(images_list)

def calculate_inception_score(images, n_split=10, eps=1E-16):
	model = load_model('cnn-mnist-model.h5')
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		subset = subset.astype('float32')
		subset = scale_images(subset, (28,28,1))
		p_yx = model.predict(subset)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		sum_kl_d = kl_d.sum(axis=1)
		avg_kl_d = mean(sum_kl_d)
		is_score = exp(avg_kl_d)
		scores.append(is_score)
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std


shuffle(X)
print('loaded', X.shape)
is_avg, is_std = calculate_inception_score(X)
print(f'Inception Score: {is_avg}\n Standart Deviation: {is_std}')