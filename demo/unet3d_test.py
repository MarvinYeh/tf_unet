import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util

plt.rcParams['image.cmap'] = 'gist_earth'

nx = 256
ny = 256
generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20, depth_3d=8)
x_test, y_test = generator(1)

net = unet.Unet3D(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, "./unet_trained", training_iters=20, epochs=10, display_step=2)
prediction = net.predict("./unet_trained/model.cpkt", x_test)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,0,...,0], aspect="auto")
ax[1].imshow(y_test[0,0,...,1], aspect="auto")
mask = prediction[0,0,...,1] > 0.9
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
fig.tight_layout()
fig.savefig("../docs/toy_problem.png")