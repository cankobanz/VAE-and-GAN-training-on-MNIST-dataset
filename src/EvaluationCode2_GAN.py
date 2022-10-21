from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

generator = load_model('generator_model_100.h5')

latent_dim = 10
n = 10
n_samples = n * n
np.random.seed(102)

rand_vec = np.random.randn(latent_dim * n_samples)
rand_vec = rand_vec.reshape(n_samples, latent_dim)
X = generator.predict(rand_vec)

plt.figure(figsize=(16, 16))
for i in range(n * n):
    plt.subplot(n, n, 1 + i)
    plt.axis('off')
    plt.imshow(X[i, :, :, 0], cmap='gray')

plt.show()
plt.close()
