#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 07:01:13 2022

@author: user01
"""
#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np
from termcolor import colored, cprint
from tqdm import trange
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpus in physical_devices:
  tf.config.experimental.set_memory_growth(gpus, True)

from models import build_discriminator, build_generator, build_gan
from data_loader import get_fake_samples, get_latent_noise, get_real_samples, load_data
from plotting import save_imgs, save_plots

#%%
def train(gen_model, desc_model, gan_model, dataset, latent_dim=1000, batch_size=128, epochs=10000, save_interval=500):
    batches_per_epoch = (dataset.shape[0] // batch_size)
    total_steps = batches_per_epoch * epochs
    half_batch = batch_size // 2

    tdr_loss, tdf_loss, tg_loss, tdr_acc, tdf_acc, x = [], [], [], [], [], []
    t = trange(total_steps, leave=True)
    for i in t:
        # genrate real samples and train disc. on real samples
        x_real, y_real = get_real_samples(dataset, half_batch)
        d_rloss, d_racc = desc_model.train_on_batch(x_real, y_real)
        # genrate  fake samples and train disc. on fake samples
        x_fake, y_fake = get_fake_samples(gen_model, latent_dim, half_batch)
        d_floss, d_facc = desc_model.train_on_batch(x_fake, y_fake)

        # Now for generator(GAN) training preparing inputs and their labels
        x_gan = get_latent_noise(latent_dim, batch_size)
        y_gan = np.ones((batch_size, 1)) # here we are fooling disc. in thinking that all samples are real. 
        g_loss = gan_model.train_on_batch(x_gan, y_gan)

        # settin visulaization and printing for observing training.
        x.append(i)
        tdf_loss.append(d_floss)
        tdr_loss.append(d_rloss)
        tdf_acc.append(d_facc)
        tdr_acc.append(d_racc)
        tg_loss.append(g_loss)
        txt1 = colored(f'DR_loss : {d_rloss:.4f}, DF_loss : {d_floss:.4f}', 'cyan')
        txt2 = colored(f'G_loss : {g_loss:.4f}', 'green')
        txt3 = colored(f'DR_acc : {100*d_racc:.01f}%, DF_acc : {100*d_facc:.01f}%', 'magenta')
        t.set_description(f'{txt1} _ {txt2} _ {txt3}')
        t.refresh()
        if (i+1) % save_interval == 0:
            save_plots(i+1, tdr_loss, tdf_loss, tdr_acc, tdf_acc, tg_loss, x)
            save_imgs(i+1, gen_model, latent_dim)

    return tdr_loss, tdf_loss, tdr_acc, tdf_acc, tg_loss
    
# %%

latentDim = 728
img_shape = (28,28,1)
Epochs = 10000
BatchSize = 128
moment = 0.5
lr = 1e-4

d_model = build_discriminator(img_shape, momentum=moment, lr=lr)
g_model = build_generator(latentDim, momentum=moment)

gan = build_gan(g_model, d_model, lr=lr)

# d_model.summary()
# g_model.summary()
# gan.summary()

print(d_model.count_params())
print(g_model.count_params())
print(gan.count_params())

data_set = load_data(batch_size=BatchSize)

_ = train(g_model, d_model, gan, data_set, latentDim, BatchSize, Epochs, save_interval=100)
# %%

noise = get_latent_noise(latentDim, 1)
imgs = g_model.predict(noise)
plt.imshow(imgs.squeeze())

# %%

from math import pi
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
# generating random data values
x = np.linspace(1, 1000, 5000)
y = np.random.randint(1, 1000, 5000)
 
# enable interactive mode
plt.ion()
 
# creating subplot and figure
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y)
 
# setting labels
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Updating plot...")
 
# looping
for _ in range(50):
   
    # updating the value of x and y
    line1.set_xdata(x*_)
    line1.set_ydata(y)
 
    # re-drawing the figure
    fig.canvas.draw()
     
    # to flush the GUI events
    fig.canvas.flush_events()
    time.sleep(10)