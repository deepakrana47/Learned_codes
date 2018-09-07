import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

X = 2*np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))
plt.plot(X)
plt.title('original')
plt.show()

decay = T.scalar('decay')
sequence = T.vector('sequence')

def rec(x, last, decay):
    return (1-decay)*x + decay*last

out, _ = theano.scan(
    fn=rec,
    sequences=sequence,
    n_steps=sequence.shape[0],
    outputs_info=[np.float64(0)],
    non_sequences=[decay],
)

sfilter = theano.function(
    inputs=[sequence, decay],
    outputs=out
)

oval = sfilter(X, decay=.99)

plt.plot(oval)
plt.title('filtered')
plt.show()