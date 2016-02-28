#!/usr/bin/env python
"""
Following:
    http://deeplearning.net/tutorial/dA.html
"""
from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class dA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize with the number of visible units (input dimension), the
        number of hidden units (dimension of the latent/hidden space), and the
        corruption level.

        * numpy_rng : numpy.random.RandomState
        * thean_rng : theano.tensor.shared_randomstreams.RandomStreams - if
        `None` is given, generate one based on a seed from `numpy_rng`
        * input: theano.tensor.TensorType - symbolic description of the input,
        or `None` for standalone dA (we pass the input so we may concatenate
        layers of autoencoders to form a deep network)
        * W: theano.tensor.TensorType - weights to be shared with the dA and
        another architecture - if dA is standalone this should be `None`
        * bhid: theano.tensor.TensorType - bias values for the hidden units
        that should be shared with another architecture - `None` if dA is
        standalone
        * bvis: theano.tensor.TensorType - bias values for the visible units
        that should be shared with another architecture - `None` if dA is
        standalone
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input is None:
            # use a matrix: expect a minibatch of examples, each in a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_corrupted_input(self, input, corruption_level):
        """
        keep `1 - corruption_level` entries of the inputs unchanged and zero
        a randomly selected subset of size `corruption_level`
        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_cost_updates(self, corruption_level, learning_rate):
        """
        compute cost and the updates for one training step of the dA
        """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # sum over the size of a datapoint - if using minibatches, `L` will be
        # a vector with one entry per minibatch example
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # now average over all examples
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
