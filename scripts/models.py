#%%
from turtle import shape
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/home/user01/data/talha/gan_1/logs/")
########################################################
#              Simple NN GAN
########################################################

# def build_generator(latent_dim, op_shape=(28,28,1), momentum=0.99, k_init=RandomNormal(stddev=0.02)):
#     # latent dimension or input noise
#     noise_shape = (latent_dim,) 
#     noise = Input(shape=noise_shape)

#     x = Dense(256, input_shape=noise_shape, kernel_initializer=k_init)(noise)
#     x = BatchNormalization(momentum=momentum)(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     x = Dense(512, kernel_initializer=k_init)(x)
#     x = BatchNormalization(momentum=momentum)(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     x = Dense(512, kernel_initializer=k_init)(x)
#     x = BatchNormalization(momentum=momentum)(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     x = Dense(1024, kernel_initializer=k_init)(x)
#     x = BatchNormalization(momentum=momentum)(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     x = Dense(np.prod(op_shape), activation='tanh', kernel_initializer=k_init)(x)
#     op = Reshape(op_shape)(x)

#     model = Model(inputs=[noise], outputs=[op])

#     return model

# def build_discriminator(img_shape, k_init=RandomNormal(stddev=0.02), lr=0.0002):

#     img = Input(shape=img_shape)

#     x = Flatten(input_shape=img_shape)(img)

#     x = Dense(512, kernel_initializer=k_init)(x)
#     #model.add(BatchNormalization())
#     x = LeakyReLU(alpha=0.2)(x)

#     x = Dense(512, kernel_initializer=k_init)(x)
#     #model.add(BatchNormalization())
#     x = LeakyReLU(alpha=0.2)(x)

#     x = Dense(256, kernel_initializer=k_init)(x)
#     #model.add(BatchNormalization())
#     x = LeakyReLU(alpha=0.2)(x)

#     confd = Dense(1, activation='sigmoid', kernel_initializer=k_init)(x)

#     model = Model(inputs=[img], outputs=[confd])

#     # compile model
#     model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5),
#                     metrics=['accuracy'])

#     return model


# def build_gan(generator, descriminator, lr=0.0002):
#     # so that it don't update while updating generator
#     descriminator.trainable = False

#     ip = generator.input
#     op = descriminator(generator.output)

#     model = Model(inputs=[ip], outputs=[op])

#     model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5))
#     return model

# def build_gan(generator, descriminator, lr=0.0002):
#     # so that it don't update while updating generator
#     descriminator.trainable = False

#     model = Sequential()
#     model.add(generator)
#     model.add(descriminator)

#     model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5))
#     return model
#%%
########################################################
#              Simple CNN GAN
########################################################
# define the standalone discriminator model
def build_discriminator(img_shape, momentum=0.8, k_init=RandomNormal(stddev=0.02), lr=0.0002):

    img_shape = Input(shape=img_shape)
    # downsample to 14x14
    x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(img_shape)
    #x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3,3), padding='same', kernel_initializer=k_init)(x)
    #x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(x)
    #x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (3,3), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    # downsample to 7x7
    x = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (3,3), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    # classifier
    x = Flatten()(x)
    op = Dense(1, activation='sigmoid', kernel_initializer=k_init)(x)

    model = Model(inputs=[img_shape], outputs=[op])
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5), metrics=['accuracy'])
    return model
 
# define the standalone generator model
def build_generator(latent_dim, op_shape=(28,28,1), momentum=0.8, k_init=RandomNormal(stddev=0.02)):

    # foundation for 7x7 image
    in_nodes = 128 * 14 * 14
    # define input
    latent_dim = Input(shape=latent_dim)
    x = Dense(in_nodes, kernel_initializer=k_init)(latent_dim)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((14, 14, 128))(x)
    # upsample to 14x14
    x = Conv2DTranspose(128, (3,3), padding='same', kernel_initializer=k_init)(x)#(4,4), strides=(2,2)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (3,3), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    # upsample to 28x28
    x = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (3,3), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(512, (3,3), padding='same', kernel_initializer=k_init)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)
    # output 28x28x1
    op = Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=k_init)(x)

    model = Model(inputs=[latent_dim], outputs=[op])
    
    return model
 
def build_gan(generator, descriminator, lr=0.0002):
    # so that it don't update while updating generator
    descriminator.trainable = False

    ip = generator.input
    op = descriminator(generator.output)

    model = Model(inputs=[ip], outputs=[op])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5))
    return model

# %%
