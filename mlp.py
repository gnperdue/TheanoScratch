#!/usr/bin/env python
"""
A multilayer perceptron using Theano

Math:
    f(x) = G(b^{(2)} + W^{(2)}(s(b^{(1)} + W^{(1)} x)))

References:
    _Pattern Recognition and Machine Learning_, Christophe Bishop, Sec. 5
"""
from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None, activation=T.tanh):
        """
        MLP hidden layer - units are fully connected and have a sigmoidal
        activation function. weight matrix `W` is of shape (n_in, n_out) and
        the bias vector `b` is of shape (n_out,)

        the nonlinearity used by default is `tanh`

        hidden unit activation by default is `tanh(dot(input, W) + b)`

        * rng - numpy.random.RandomState - a random number generator for weight
        initialization

        * input - theano.tensor.dmatrix - symbolic tensor of shape
        (n_examples, n_in)

        * n_in - int - dimensionality of the input

        * n_out - int - number of hidden units

        * activation - theano.Op or function - non-linearity to be applied to
        the hidden layer
        """
        self.input = input

        # `W` is initalized with `W_values` which is uniformly sampled from
        # [-sqrt(6./(n_in + n_hidden)), sqrt(6./(n_in + n_hidden))] for a
        # `tanh` activation function.
        #
        # the output of unifrom is converted using `asarray` to dtype
        # theano.config.floatX so the code may be run on a GPU
        #
        # note: optimal initialization is dependent on (among other things) the
        # activation function used
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6.0 / (n_in + n_out)),
                    high=numpy.sqrt(6.0 / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """
    multi-layer perceptron with _one_ hidden layer
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """
        initialize the params of the mlp

        rng - numpy.random.RandomState - a random number generator used to
        initialize weights

        input - T.TensorType - symbolic var that describes the input of the
        architecture (one minibatch)

        n_in - int - number of input units (dimension of the space of the
        datapoints)

        n_hidden - int - number of hidden units

        n_out - int - number of output units (dimension of the space of the
        labels)
        """
        # this is a one-hidden-layer MLP, so we will create a HiddenLayer
        # with `tanh` activation connected to the logistic regression layer.
        # the activation function can be replaced by a sigmoid (or something
        # else)
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # the logistic regression layer gets as input the hidden units of the
        # hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm - one regularization option is to enforce L1 norm be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum() +
            abs(self.logRegressionLayer.W).sum()
        )

        # L2 norm - one regularization option is to enforce L2 norm be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum() +
            (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log-likelihood of the MLP is given by the negative LL of the
        # output of the model, computed in the logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        # same for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the params of the model are the params of the two layers composing it
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    stochastic gradient descent optimization for a MLP using MNIST

    learning_rate - float - rate factor for sgd

    L1_reg - float - L1-norm's weight when added to the cost

    l2_reg - float - L2-norm's weight when added to the cost

    n_epochs - int - maximal number of epochs to run the optimizer

    dataset - string - path of the MNIST data
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for each stage
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print("...building the model")

    # symbolic vars for the data
    index = T.lscalar()      # minibatch index
    x = T.matrix('x')        # rasterized image data
    y = T.ivector('y')       # labels 0,1,...,9

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); the cost is
    # expressed here symbolically
    cost = (
        classifier.negative_log_likelihood(y) +
        L1_reg * classifier.L1 +
        L2_reg * classifier.L2_sqr
    )

    # theano function that computes the mistakes made by the model over a
    # minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of the cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update_expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compile a Theano function to return the cost and update the parameters of
    # the model based on the rules defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print("...training")

    # early-stopping parameters
    patience = 10000       # min. number of examples
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1

        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                # zero-one loss on the validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print("epoch %i, minibatch %i/%i, minibatch avg cost %f, "
                      "validation err %f %%" %
                      (epoch, minibatch_index + 1, n_train_batches,
                       minibatch_avg_cost, this_validation_loss * 100.0))

                # if this is our best validation score so far...
                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < \
                            best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter_num

                    # test on the test set
                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print("   epoch %i, minibatch %i/%i, test error of "
                          "best model %f %%" %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.0))

            if patience < iter_num:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print("Optimization complete. Best validation score of %f %% "
          "obtained at iteration %i, with test performance %f %%" %
          (best_validation_loss * 100.0, best_iter + 1, test_score * 100.0))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.0))


if __name__ == '__main__':
    test_mlp()
