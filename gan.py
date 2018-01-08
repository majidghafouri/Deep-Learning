# Majid.Ghafouri@Gmail.com
# January 7, 2018
# Preparing the MNSIT dataset

from keras.datasets import mnist
# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')

# Next, we will randomize the handwritten digits by using numpy to
# create random permutations on the dataset on the rows (images)

import numpy as np

# #Use a seed so that we get the same random permutation each time
# np.random.seed(1)
# p = np.random.permutation(mnist.data.shape[0])
# X = mnist.data[p]

# reshape the dataset from 70000x786 to 70000x28x28, so that every
# image in the dataset is arranged into a 28x28 grid,
# each cell in the grid represents 1 pixel of the image.

# X = X.reshape((70000, 28, 28))

# -----------------------------------

img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# Since the DCGAN that we’re creating takes in a 64x64 image as the input,
# we will use numpy to resize the each 28x28 image to 64x64 images:
X = X_train[1:100, :, :]
X = np.resize(X, (X.shape[0], 64, 64))

# Each pixel in the 64x64 image is represented by a number between 0-255,
# that represents the intensity of the pixel. However, we want to input numbers
# between -1 and 1 into the DCGAN

X = X.astype(np.float32) / (255.0 / 2) - 1.0

# Ultimately, images are fed into the neural net through a 70000x3x64x64 array but
# they are currently in a 70000x64x64 array. We need to add 3 channels to the images.
# Typically, when we are working with the images, the 3 channels represent the red, green,
# and blue (RGB) components of each image. Since the MNIST dataset is grayscale,
# we only need 1 channel to represent the dataset. We will pad the other channels with 0’s:

X = X.reshape((X.shape[0], 1, 64, 64))
X = np.tile(X, (1, 3, 1, 1))

# MXNet to easily iterate through the images during training
import mxnet as mx

batch_size = 10
image_iter = mx.io.NDArrayIter(X, batch_size=batch_size)


# Preparing Random Numbers
# We use MXNet’s built-in mx.random.normal function to return the random numbers
# from a normal distribution during the iteration.

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        # Returns random numbers from a gaussian (normal) distribution
        # with mean=0 and standard deviation = 1
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

    # When we initialize the RandIter, we need to provide two numbers: the batch size
    #  and how many random numbers we want in order to produce a single image from.
    # This number is referred to as Z, and we will set this to 100.
    # Every time we iterate and get a batch of random numbers,
    # we will get a 4 dimensional array with shape


Z = 100
rand_iter = RandIter(batch_size, Z)

# Create the Model
# The Generator

no_bias = True
fix_gamma = True
epsilon = 1e-5 + 1e-12

rand = mx.sym.Variable('rand')

g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4, 4), num_filter=1024, no_bias=no_bias)
gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=epsilon)
gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=512, no_bias=no_bias)
gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=epsilon)
gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=256, no_bias=no_bias)
gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=epsilon)
gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=128, no_bias=no_bias)
gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=epsilon)
gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3, no_bias=no_bias)
generatorSymbol = mx.sym.Activation(g5, name='gact5', act_type='tanh')

# Create the Model
# The Discriminator

data = mx.sym.Variable('data')

d1 = mx.sym.Convolution(data, name='d1', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=128, no_bias=no_bias)
dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=256, no_bias=no_bias)
dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=epsilon)
dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=512, no_bias=no_bias)
dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=epsilon)
dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=1024, no_bias=no_bias)
dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=epsilon)
dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4, 4), num_filter=1, no_bias=no_bias)
d5 = mx.sym.Flatten(d5)

label = mx.sym.Variable('label')
discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')

# Prepare the models using the Module API

# Hyper-parameters
sigma = 0.02
lr = 0.0002
beta1 = 0.5
# If you do not have a GPU. Use the below outlined
ctx = mx.cpu()
# ctx = mx.gpu(0)

# =============Generator Module=============
generator = mx.mod.Module(symbol=generatorSymbol, data_names=('rand',), label_names=None, context=ctx)
generator.bind(data_shapes=rand_iter.provide_data)
generator.init_params(initializer=mx.init.Normal(sigma))
generator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods = [generator]

# =============Discriminator Module=============
discriminator = mx.mod.Module(symbol=discriminatorSymbol, data_names=('data',), label_names=('label',), context=ctx)
discriminator.bind(data_shapes=image_iter.provide_data,
                   label_shapes=[('label', (batch_size,))],
                   inputs_need_grad=True)
discriminator.init_params(initializer=mx.init.Normal(sigma))
discriminator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods.append(discriminator)

# Visualizing The Training

from matplotlib import pyplot as plt


# Takes the images in the batch and arranges them in an array so that they can be
# Plotted using matplotlib
def fill_buf(buf, num_images, img, shape):
    width = buf.shape[0] / shape[1]
    height = buf.shape[1] / shape[0]
    img_width = int(num_images % width) * shape[0]
    img_hight = int(num_images / height) * shape[1]
    buf[img_hight:img_hight + shape[1], img_width:img_width + shape[0], :] = img


# Plots two images side by side using matplotlib
def visualize(fake, real):
    # 64x3x64x64 to 64x64x64x3
    fake = fake.transpose((0, 2, 3, 1))
    # Pixel values from 0-255
    fake = np.clip((fake + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
    # Repeat for real image
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)

    # Create buffer array that will hold all the images in the batch
    # Fill the buffer so to arrange all images in the batch onto the buffer array
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros((int(n * fake.shape[1]), int(n * fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fill_buf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros((int(n * real.shape[1]), int(n * real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fill_buf(rbuff, i, img, real.shape[1:3])

    # Create a matplotlib figure with two subplots: one for the real and the other for the fake
    # fill each plot with the buffer array, which creates the image
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(fbuff)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(rbuff)
    plt.show()


# Fit the Model

# =============train===============
print('Training...')
for epoch in range(1):
    image_iter.reset()
    for i, batch in enumerate(image_iter):
        # Get a batch of random numbers to generate an image from the generator
        rbatch = rand_iter.next()
        # Forward pass on training batch
        generator.forward(rbatch, is_train=True)
        # Output of training batch is the 64x64x3 image
        outG = generator.get_outputs()

        # Pass the generated (fake) image through the discriminator, and save the gradient
        # Label (for logistic regression) is an array of 0's since this image is fake
        label = mx.nd.zeros((batch_size,), ctx=ctx)
        # Forward pass on the output of the discriminator network
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        # Do the backward pass and save the gradient
        discriminator.backward()
        gradD = [[grad.copyto(grad.context) for grad in grads] for grads in discriminator._exec_group.grad_arrays]

        # Pass a batch of real images from MNIST through the discriminator
        # Set the label to be an array of 1's because these are the real images
        label[:] = 1
        batch.label = [label]
        # Forward pass on a batch of MNIST images
        discriminator.forward(batch, is_train=True)
        # Do the backward pass and add the saved gradient from the fake images to the gradient
        # generated by this backwards pass on the real images
        discriminator.backward()
        for gradsr, gradsf in zip(discriminator._exec_group.grad_arrays, gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        # Update gradient on the discriminator
        discriminator.update()

        # Now that we've updated the discriminator, let's update the generator
        # First do a forward pass and backwards pass on the newly updated discriminator
        # With the current batch
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        discriminator.backward()
        # Get the input gradient from the backwards pass on the discriminator,
        # and use it to do the backwards pass on the generator
        diffD = discriminator.get_input_grads()
        generator.backward(diffD)
        # Update the gradients on the generator
        generator.update()

        # Increment to the next batch, printing every 50 batches
        i += 1
        # if i % 1 == 0:
        print('epoch:', epoch, 'iter:', i)
        print
        print("   From generator:        From MNIST:")
        visualize(outG[0].asnumpy(), batch.data[0].asnumpy())
