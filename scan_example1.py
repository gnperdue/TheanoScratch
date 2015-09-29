#!/usr/bin/env python
"""
Following:
    http://deeplearning.net/software/theano/tutorial/loop.html

Scan Example: computing `tanh(x(t).dot(W) + b` elementwise
"""
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np

# define the tensor variables
X = T.matrix('X')
W = T.matrix('W')
b_sym = T.vector('b_sym')

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym),
                               sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym],
                                      outputs=[results])

# test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

print(compute_elementwise(x, w, b)[0])

# comparison with numpy
print(np.tanh(x.dot(w) + b))
