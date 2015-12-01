#!/usr/bin/env python
"""
A convolutional neural net example using Theano.

Usage:
    python convolutional_mlp.py
                  [-t/--train]
                  [-p/--predict]
                  [-d/--data <path-to-dataset>]
                  [-n/--nepochs <# of epochs>]
                  [-r/--rate <learning rate>]
                  [-b/--batch <batch size>]

    Default train False
            predict False
            dataset: "../Datasets/mnist.pkl.gz"
            N epochs: 1000
            batch size: 500
            rate: 0.01

References:
    * http://deeplearning.net/tutorial/lenet.html
    * Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
      Gradient-Based Learning Applied to Document
      Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
        http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""
from __future__ import print_function
import os
# import sys
import timeit
import cPickle

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


BEST_PICKLEJAR = 'convolutional_mlp_best_model.pkl'


class LeNetConvPoolLayer(object):
    """
    Pool layer of a convolutional neural network.
    """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),
                 W=None, b=None):
        """
        LeNetConvPoolLayer with shared variable internal parameters.

        rng - numpy.random.RandomState - random number generator to initialize
        weights

        input - theano.tensor.dtensor4 - symbolic image tensor of shape
        image_shape

        filter_shape - tuple or list of length 4 -
            (# of filters,
             # of input feature maps,
             filt. height,
             filt. wid)

        image_shape - tuple or list of length 4 -
            (batch size,
             # of input feature maps,
             img height,
             img wid)

        poolsize - tuple or list of length 2 - the downsampling (pooling)
        factor (#rows, #cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        if W is None:
            # there are `# input feat. maps * filt h * filt w` inputs to each
            # hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer gets a gradient from:
            #  `# output feat maps * filt h * filt w / pooling size`
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                       numpy.prod(poolsize))
            # init with random weights
            W_bound = numpy.sqrt(6.0 / (fan_in + fan_out))
            W_values = numpy.asarray(
                rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=filter_shape
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            # bias is a 1d tensor with one bias output per feat. map
            b_values = numpy.zeros((filter_shape[0],),
                                   dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term - we must reshape the 1d tensor into shape
        # (1, n_filters, 1, 1) - this means each bias will be broadcast across
        # minibatches and feat. map h and w.
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store the params of the layer
        self.params = [self.W, self.b]

        # model input
        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """
    demo LeNet on the MNIST dset

    learning_rate - float - factor for stochastic gradient

    n_epochs - int - max # of epochs to run the optimizer

    dataset - string - path to the dset

    nkerns - [ints] - number of kernels on each layer

    batch_size - int - self describing!
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # num of minibatches for training, validation, testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # symbolic vars for the data
    index = T.lscalar()     # minibatch index
    x = T.matrix('x')       # data is rasterized images
    y = T.ivector('y')      # labels are 1D [0,1...9]

    print("... building the model")

    # Reshape matrix of rasterized images of shape (batch_size, 28*28) to a
    # 4D tensor, compatible with the LeNetConvPoolLayer. (28, 28) is the size
    # of MNIST images
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28 - 5 + 1, 28 - 5 + 1) = (24, 24),
    # maxpooling reduces this further to (24 / 2, 24 / 2) = (12, 12).
    # 4D output tensor is therefore of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer; filtering reduces the
    # image size to (12 - 5 + 1, 12 - 5 + 1) = (8, 8) and maxpooling further
    # reduces it to (8 / 2, 8 / 2) = (4, 4); 4D output tensor therefore has
    # shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # The HiddenLayer is fully connected, so it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e., the matrix of rasterized images).
    # This generates a martix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(
        input=layer2.output,
        n_in=500,
        n_out=10
    )

    # the cost we minimize is the negative log-likelihood (NLL) of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # `train_model()` updates model parameters via SGD
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training')

    # early-stopping params
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and not done_looping:

        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training at iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, cost %f, '
                      'validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       cost_ij, this_validation_loss * 100.0))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') % (epoch, minibatch_index + 1,
                                             n_train_batches,
                                             test_score * 100.0))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, with test '
          'performance %f %%' % (best_validation_loss * 100.0,
                                 best_iter + 1, test_score * 100.0))
    print('The code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.0))

    # save the best model - can't seem to pickle the whole model here, rather
    # than debug, let's just save the params and use them to re-create the
    # model
    with open(BEST_PICKLEJAR, 'w') as f:
        cPickle.dump(params, f)


def predict(dataset, nkerns=[20, 50]):
    """
    example of loading and running a model
    """
    # test on some examples from the test set
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    pars = cPickle.load(open(BEST_PICKLEJAR))

    # load the saved weights and bias vectors
    for i, p in enumerate(pars):
        print("Checking loaded parameter %i type and shape..." % i)
        print(type(p))
        print(p.eval().shape)

    # params = layer3.params + layer2.params + layer1.params + layer0.params
    # pars[0] -> logreg W (layer "3")
    # pars[1] -> logreg b (layer "3")
    # pars[2] -> hidden W (layer "2")
    # pars[3] -> hidden b (layer "2")
    # pars[4] -> conv pool layer 1 W
    # pars[5] -> conv pool layer 1 b
    # pars[6] -> conv pool layer 0 W
    # pars[7] -> conv pool layer 0 b

    # symbolic vars for the data
    rng = numpy.random.RandomState(23455)  # don't actually use this...
    x = T.matrix('x')       # data is rasterized images

    # use this parameter to control the chunk of the data we evaluate
    # could set it to 1 and scan the predict function (see below) in a loop
    # instead perhaps
    samp_size = 20

    layer0_input = x.reshape((samp_size, 1, 28, 28))
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        # input=x.reshape((1, 1, 28, 28)),
        image_shape=(samp_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2),
        W=pars[6],
        b=pars[7]
    )
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(samp_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2),
        W=pars[4],
        b=pars[5]
    )
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh,
        W=pars[2],
        b=pars[3]
    )
    layer3 = LogisticRegression(
        input=layer2.output,
        n_in=500,
        n_out=10,
        W=pars[0],
        b=pars[1]
    )

    # compile a predictor fn
    predict_model = theano.function(
        inputs=[layer0.input],
        outputs=layer3.y_pred
    )

    predicted_values = \
        predict_model(test_set_x[:samp_size].reshape(samp_size, 1, 28, 28))
    print("Predicted values for the first %d:" % (samp_size))
    print(predicted_values)
    print("Actual values:")
    print(T.cast(test_set_y, 'int32').eval()[:samp_size])


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data', dest='dataset',
                      default='../Datasets/mnist.pkl.gz', help='Data set',
                      metavar='DATASET')
    parser.add_option('-n', '--nepochs', dest='n_epochs', default=200,
                      help='Number of epochs', metavar='N_EPOCHS',
                      type='int')
    parser.add_option('-r', '--rate', dest='lrate', default=0.01,
                      help='Learning rate', metavar='LRATE',
                      type='float')
    parser.add_option('-b', '--batch', dest='batch_size', default=500,
                      help='Batch size', metavar='BSIZE',
                      type='int')
    parser.add_option('-t', '--train', dest='do_train', default=False,
                      help='Run the training', metavar='DO_TRAIN',
                      action='store_true')
    parser.add_option('-p', '--predict', dest='do_predict', default=False,
                      help='Run a prediction', metavar='DO_PREDICT',
                      action='store_true')
    (options, args) = parser.parse_args()

    if not options.do_train and not options.do_predict:
        print("\nMust specify at least either train or predict:\n\n")
        print(__doc__)

    n_kerns = [20, 50]

    if options.do_train:
        evaluate_lenet5(learning_rate=options.lrate,
                        n_epochs=options.n_epochs,
                        dataset=options.dataset,
                        batch_size=options.batch_size,
                        nkerns=n_kerns)

    if options.do_predict:
        predict(dataset=options.dataset,
                nkerns=n_kerns)
