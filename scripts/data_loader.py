import numpy as np
from tensorflow.keras.datasets import mnist

def load_data(batch_size=128):
    (X_train, Y_train), (_, _) = mnist.load_data()

    # Rescale -1 to 1 
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #from 28*28 to 28*28*1
    X_train = np.expand_dims(X_train, axis=3) 
    
    # idx = Y_train == 8
    # X_train = X_train[idx]

    return X_train

'''half_batch will be referred to as n_samples'''

def get_real_samples(dataset, n_samples):
    # get random ids
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    # get images
    imgs = dataset[idx]
    # also generate their labels
    lbls = np.ones((n_samples, 1))

    return imgs, lbls

def get_fake_samples(generator, latent_dim, n_samples):

    noise = get_latent_noise(latent_dim, n_samples)
    imgs = generator.predict(noise)
    # as these are fake images so just make -ve labels
    lbls = np.zeros((n_samples, 1))
    return imgs, lbls

def get_latent_noise(latent_dim, n_samples):
    return np.random.normal(0, 1, (n_samples, latent_dim))