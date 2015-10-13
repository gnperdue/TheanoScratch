#!/usr/bin/env python
"""
Following:
    http://deeplearning.net/tutorial/logreg.html
"""
from __future__ import print_function
import cPickle
import gzip
import os
import timeit

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """
    Multi-class Logistic Regression
    """
    def __init__(self, input, n_in, n_out, W=None, b=None):
        """
        input - T.TensorType - symbolic var for the input (one minibatch)

        n_in - int - dimension of data space (e.g. for 28x28 pixel images,
        it is 784)

        n_out - int - dimension of labels (e.g., for 0,1...9, it is 10)
        """
        # init with weights & bias at 0 if we aren't handed them
        if W is None:
            W_values = numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        if b is None:
            b_values = numpy.zeros(
                (n_out),
                dtype=theano.config.floatX
            )
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.b = b

        # symbolic exp for computing class-membership probabilities
        # * W is a matrix where column k represents the separation hyperplane
        #   for class k (so it is n_in rows by n_out columns)
        # * x (input) is a matrix where row-j represents the input training
        #   sample j (each such row-vector has n_in elements)
        # * b is a vector where element k represents the free parameter of
        #   hyperplane k
        # here the dot of (input, W) has n_sample rows by n_classes columns,
        # and b is a vector of n_classes in length that is added to each row
        # in the output of the dot of (input, W)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic desc. for class by max probability
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # params of the model
        self.params = [self.W, self.b]

        # keep track of the model input
        self.input = input

    def negative_log_likelihood(self, y):
        """
        compute the symbolic loss for a given minibatch

        y - T.TensorType - vector of labels
        """
        # y.shape[0] is the number of rows in y - i.e., the number of examples
        # in the minibatch. note that we use `shape[0]` to pick out the
        # number - y.shape will have a value like (n,), so y.shape[0] is n.
        #
        # T.arange is a symbolic vector containing [0, 1, 2, ..., n-1].
        #
        # T.Log(self.p_y_given_x) is a matrix of log-probabilities with one
        # row per example and one column per class.
        #
        # using the
        # [T.arange(y.shape[0]), y] notation picks out the vector
        # [logp[0, y[0]], logp[1, y[1]], logp[2, y[2]], ..., logp[n-1, y[n-1]].
        # note that we take the mean of this vector even though formally the
        # math calls for the sum to reduce dependence on minibatch size during
        # training.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch over
        the total number of examples in that minibatch - i.e., zero-one loss
        over the size of the minibatch

        y - T.TensorType - vector of labels
        """
        # check for matching dimensions between labels and predictions
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))

        # check the datatype of the labels
        if y.dtype.startswith('int'):
            # .neq returns a vector of 0s and 1s where 1 represents a failure
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    """
    dataset - string - path to the data (here MNIST)
    """
    # download the data if we don't have it
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "Datasets",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print("Downloading data from %s" % origin)
        urllib.urlretrieve(origin, dataset)

    print("...loading data")

    f = gzip.open(dataset, 'rb')
    # these datasets are in format - tuple(input, target), where the input
    # is 2d numpy.ndarray and target is a vector (with length equal to the
    # number of rows in input) of labels
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """
        load the dataset into shared variables for efficient copies to a GPU
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # data stored on the GPU must be stored as floats - shared_y will do
        # that. but, for computations, we need them as ints, so we'll cast
        # them as such here (it won't affect shared_y)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13,
                           n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    stochastic gradient descent optimization of a log-linear model. here we
    operate on MNSIT.

    learning_rate - float - rate factor in SGD

    n_epochs - int - max number of epochs to run the optimizer

    dataset - string - path to the MNIST data

    batch_size - int
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # number of minibatches for each stage
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # build the model
    print("...building the model")

    # symbolic vats for the data
    index = T.lscalar()       # minibatch idx
    x = T.matrix('x')         # rasterized image data, 28x28 pixels
    y = T.ivector('y')        # labels 0,1..9

    # logistic regression class
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    # cost we minimize
    cost = classifier.negative_log_likelihood(y)

    # compile fns that compute mistakes on a minibatch
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

    # gradients of cost w.r.t. theta=(W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update params via grad descent
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # fn to get the cost, but also update params
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # run the training
    print("...training the model")

    # early stopping params
    patience = 5000            # min number of examples to consider
    patience_increase = 2      # wait this much longer if signif. improve.
    improvement_threshold = 0.95
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.0
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1

        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                # zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation err %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.0))

                if this_validation_loss < best_validation_loss:
                    # improve patience if improvement is significant
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss

                    # test on the test set
                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print('epoch %i, minibatch %i/%i, test err %f %%' %
                          (epoch, minibatch_index + 1, n_test_batches,
                           test_score * 100.0))

            if patience <= iter_num:
                done_looping = True
                break

    # save the best model
    with open('best_model.pkl', 'w') as f:
        cPickle.dump(classifier, f)

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%,'
          'with test performance %f %%' % (best_validation_loss * 100.0,
                                           test_score * 100.0))
    print('The code ran for %d epochs, with %f epochs/sec' %
          (epoch, 1.0 * epoch / (end_time - start_time)))


def predict():
    """
    example of loading and running a model
    """
    classifier = cPickle.load(open('best_model.pkl'))

    # compiler a predictor fn
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    # test on some examples from the test set
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10:")
    print(predicted_values)
    print("Actual values:")
    print(T.cast(test_set_y, 'int32').eval()[:10])


if __name__ == '__main__':
    sgd_optimization_mnist()
    predict()
