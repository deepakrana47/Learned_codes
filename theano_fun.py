import numpy as np
import theano, theano.tensor as T
from theano import pp

x = T.scalar('x')

f = 2*x**3 + 3* x**2 + 5*x + 12

gt = T.grad(f, x)

fun = theano.function(inputs=[x], outputs=f)
diff = theano.function(inputs=[x], outputs=gt)

a = 2
print("input:",a)
print(pp(fun))
print("function value:",fun(a))
print(pp(diff))
print("function diff value:",diff(a))