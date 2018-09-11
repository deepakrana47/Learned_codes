import theano, theano.tensor as T
import numpy as np
from util import init_weight, all_parity_pairs
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class HiddenLayer:

    def __init__(self, M1, M2, id):
        self.M1 = M1
        self.M2 = M2
        self.id = id
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)

class mlp:
    def __init__(self, hidden_layer_sz):
        self.hidden_layer_sz = hidden_layer_sz

    def fit(self, X, Y, learning_rate=10e-1, mu=.99, reg=1.0, epochs=100, batch_sz=100, print_period=10, show_fig=True):
        Y = Y.astype(np.int32)
        D = X.shape[1]
        O = len(set(Y))
        M1 = D

        self.hidden_layers = []
        count = 0
        for M2 in self.hidden_layer_sz:
            self.hidden_layers.append(HiddenLayer(M1, M2, count))
            M1 = M2

        W = init_weight(M1, O)
        b = np.zeros(O)

        self.Wo = theano.shared(W, "out_weight")
        self.bo = theano.shared(b, "out_bais")

        self.params = [self.Wo, self.bo]
        for hlayer in self.hidden_layers:
            self.params += hlayer.params


        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        rcost = reg * T.sum([(p * p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
                (p, p + mu * dp - learning_rate * g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
                (dp, mu * dp - learning_rate * g) for dp, g in zip(dparams, grads)
        ]

        train_op = theano.function(
            inputs = [thX,thY],
            outputs = [cost, prediction],
            updates = updates,
        )

        n_batches = int(len(X)/batch_sz)
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Y[j * batch_sz:(j * batch_sz + batch_sz)]

                c, p = train_op(Xbatch, Ybatch)

                if j % print_period == 0:
                    costs.append(c)
                    e = np.mean(Ybatch != p)
                    print("i:",i,"j:",j,"nb:",n_batches,"cost:",c,"error:",e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.Wo) + self.bo)

    def predict(self, X):
        pX = self.forward(X)
        return T.argmax(pX, axis=1)

def wide():
    X,Y = all_parity_pairs(12)
    model = mlp([512, 265])
    model.fit(X, Y, learning_rate=10e-5, print_period=10, epochs=300, show_fig=True)

if __name__=='__main__':
    wide()
    # deep()