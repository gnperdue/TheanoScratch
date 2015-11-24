#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

x = T.vector('x')
y = T.vector('y')
val = T.dot(x, y)
f = theano.function(inputs=[x, y], outputs=val)

# check architecture
if any([xx.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm']
        for xx in f.maker.fgraph.toposort()]):
    print('Used the cpu')
elif any([xx.op.__class__.__name__ in ['GpuGemm', 'GpuGemv']
          for xx in f.maker.fgraph.toposort()]):
    print('Used the gpu')
else:
    print('ERROR, not able to tell if Theano used the cpu or gpu')
    print(f.maker.fgraph.toposort())

v1 = np.array([1.0, 2.0, 3.0], dtype=theano.config.floatX)
v2 = np.array([1.0, 2.0, 3.0], dtype=theano.config.floatX)

print(f(v1, v2))
