import numpy as np
import theano
import theano.tensor as T

# x = T.vector('x')
N=T.iscalar('N')

def fib(N, fn1, fn2):
    return fn1 + fn2, fn1

out, updates = theano.scan(
    fn=fib,
    sequences=T.arange(N),
    n_steps=N,
    outputs_info=[1., 1.]
)

fibonacci = theano.function(
    inputs=[N],
    outputs= out,
)

o = fibonacci(10)

print(o)