
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
#plt.ion()
import numpy as np

def save_imgs(epoch, gen_model, latent_dim):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = gen_model.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c, figsize = (3,3))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    
    fig.savefig(f"/home/user01/data/talha/gan_1/preds/mnist_{epoch}.png")
    plt.show()
    #plt.close()    

    return None#fig

def save_plots(epoch, dr_loss, df_loss, dr_acc, df_acc, g_loss, x):

    #f, axe = plt.subplots(figsize = (5,3))
    f, axs = plt.subplots(3, 1, figsize = (6,3))
    axs[0].plot(x, dr_loss, 'b', label='d-real')
    axs[0].plot(x, df_loss, 'g-.', label='d-fake')
    axs[0].get_xaxis().set_visible(False)
    axs[0].legend()
    # # plot discriminator accuracy
    axs[1].plot(x, dr_acc, 'r', label='acc-real')
    axs[1].plot(x, df_acc, 'c-.', label='acc-fake')
    axs[1].get_xaxis().set_visible(False)
    axs[1].legend()

    axs[2].plot(x, g_loss, 'm', label='gen')
    axs[2].legend()

    # save plot to file
    f.savefig(f"/home/user01/data/talha/gan_1/plots/mnist_{epoch}.png")
    plt.show()
    #plt.close()
       

    return None#f

#def summary_printer():

# %%



# x = [1,2,3,4,5,6,7,8]
# y=  [2,5,7,6,9,4,25,30]

# f, axs = plt.subplots(3, 1, figsize = (6,3))
# axs[0].plot(x, y, 'b', label='d-real')
# axs[0].plot(x, x, 'g-.', label='d-fake')
# axs[0].get_xaxis().set_visible(False)
# #axs[0].get_yaxis().set_visible(True)
# axs[0].legend()
# # # plot discriminator accuracy
# axs[1].plot(x, y, 'r', label='acc-real')
# axs[1].plot(x, x, 'c-.', label='acc-fake')
# axs[1].get_xaxis().set_visible(False)
# axs[1].legend()

# axs[2].plot(x, x, 'm', label='gen')
# axs[2].legend()
# plt.show()