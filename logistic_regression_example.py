#!/usr/bin/env python
"""
See:
    http://deeplearning.net/software/theano/tutorial/examples.html
    http://deeplearning.net/software/theano/tutorial/modes.html
"""
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

# theano.config.floatX = 'float32'

rng = np.random

N = 400
feats = 784
D = (rng.randn(N, feats).astype(theano.config.floatX),
     rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
training_steps = 10000

# declare Theano symbolic variables
x = T.matrix('x')
y = T.vector('y')
w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name='w')
b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name='b')
print("Initial model:")
print(w.get_value())
print(b.get_value())

# construct the Theano expression graph using the symbolic vars
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))            # P(target = 1)
prediction = p_1 > 0.5                             # prediction threshold
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)  # cross-entropy loss func.
cost = xent.mean() + 0.01 * (w ** 2).sum()         # cost func. to minimize
gw, gb = T.grad(cost, [w, b])                      # gradient of cost

# compile functions from the expression graph (optimize the graph)
# note that we don't need to pass the shared `w` and `b` in as inputs
# to either function
train = theano.function(
    inputs=[x, y],
    outputs=[prediction, xent],
    updates=[(w, w - 0.1 * gw), (b, b - 0.1 * gb)],
    name='train'
)
predict = theano.function(inputs=[x], outputs=prediction, name='predict')

# check architecture
if any([xx.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm']
        for xx in train.maker.fgraph.toposort()]):
    print('Used the cpu')
elif any([xx.op.__class__.__name__ in ['GpuGemm', 'GpuGemv']
          for xx in train.maker.fgraph.toposort()]):
    print('Used the gpu')
else:
    print('ERROR, not able to tell if Theano used the cpu or gpu')
    print(train.maker.fgraph.toposort())

# train - shared `w` and `b` updated automatically
for _ in xrange(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
