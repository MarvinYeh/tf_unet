
# coding: utf-8

# In[1]:


from __future__ import division, print_function
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['image.cmap'] = 'gist_earth'


# In[2]:


from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util


# In[3]:


nx = 572
ny = 572


# In[4]:


generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)


# In[5]:


x_test, y_test = generator(1)


# In[6]:


fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")


# In[7]:


net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)


# In[8]:


trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))


# In[9]:


path = trainer.train(generator, "./unet_trained", training_iters=20, epochs=10, display_step=2)


# In[22]:


x_test, y_test = generator(1)

prediction = net.predict("./unet_trained/model.cpkt", x_test)


# In[23]:


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
mask = prediction[0,...,1] > 0.9
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
fig.tight_layout()
fig.savefig("../docs/toy_problem.png")


# In[ ]:




