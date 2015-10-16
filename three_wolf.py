#!/usr/bin/env python
"""
This is Theano code for implementing a convolutional layer. The input consists
of 3 feature maps (an RGB color image) of size 120x160. We use convolutional
filters with 9x9 receptive fields.

Following:
    http://deeplearning.net/tutorial/lenet.html
"""

import theano
from theano import tensor as T
from theano.tensor.nnet import conv

import numpy
import pylab
from PIL import Image

rng = numpy.random.RandomState(23455)

# 4D tensor for input
input = T.tensor4(name='input')

# shared variable for weights
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared(
    numpy.asarray(
        rng.uniform(
            low=-1.0 / w_bound,
            high=1.0 / w_bound,
            size=w_shp
        ),
        dtype=input.dtype
    ),
    name='W'
)

# initialize shared variables for bias with random values
# NOTE: biases are typically initialized to zero - however, here we simply
# apply the convolutional layer to an image without learning its parameters.
# therefore, we initialize them to random values to "simulate" learning.
b_shp = (2,)
b = theano.shared(
    numpy.asarray(
        rng.uniform(
            low=-0.5,
            high=0.5,
            size=b_shp
        ),
        dtype=input.dtype
    ),
    name='b'
)

# symbolic expression for the convolution of input with filters in w
conv_out = conv.conv2d(input, W)

# symbolic expression to add bias and apply activation function - i.e.,
# produce the neural network output
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# Some comments on `dimshuffle`:
# `dimshuffle` is useful for reshaping tensors - it lets us shuffle the
# dimension around, and also to insert new ones along which the tensor will be
# broadcastable:
#  dimshuffle('x', 2, 'x', 0, 1)
#  - this will work on 3d tensors with no broadcastable dimensions. the first
#    dimension will be broadcastable, then we will have the third dimension of
#    the input tensor as the second of the resulting tensor, etc. If the tensor
#    has shape (20, 30, 40), the resulting tensor will have dimensions
#    (1, 40, 1, 20, 30) - so (AxBxC) is mapped to (1xCx1xAxB).
# More examples:
#  dimshuffle('x') - make a 0d (scalar) into a 1d vector
#  dimshuffle(0, 1) - identity
#  dimshuffle(1, 0) - inverts the first and second dimensions
#  dimshuffle('x', 0) - make a row out of a 1d vector (N to 1xN)
#  dimshuffle(0, 'x') - make a column out of a 1d vector (N to Nx1)
#  dimshuffle(2, 0, 1) - AxBxC to CxAxB
#  dimshuffle(0, 'x', 1) - AxB to Ax1xB
#  dimshuffle(1, 'x', 0) - AxB to Bx1xA

# function to compute filtered images
f = theano.function([input], output)

# open image of dimensions 639x516
img = Image.open(open('3wolfmoon.jpg'))

# dimensions are (height, width, channel)
# so, `img.shape` = (639, 516, 3), etc.
img = numpy.asarray(img, dtype='float64') / 256.0
height, width, channel = img.shape

# put image into 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, channel, height, width)

filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1)
pylab.axis('off')
pylab.imshow(img)
# recall the convOp output (filtered image) is actually a "minibatch" of size 1
# here, so we take index 0 in the first dimension
pylab.subplot(1, 3, 2)
pylab.axis('off')
pylab.imshow(filtered_img[0, 0, :, :])
#
pylab.subplot(1, 3, 3)
pylab.axis('off')
pylab.imshow(filtered_img[0, 0, :, :])
pylab.show()
