import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import init_weight, all_parity_pairs_with_sequence_labels
from sklearn.utils import shuffle


class SimpleRNN:
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, learning_rate=10e-1, mu=.99, reg=1.0, activation=T.tanh, batch_sz=100, epochs=100,
            show_fig=False):
        D = X[0].shape[1]
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.f = activation

        # initial weights
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.Wo, self.bo, self.h0]

        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        thStartPoints = T.ivector('startPoints')

        XW = thX.dot(self.Wx)

        def recurrence(xw_t, is_start, h_t1, h0):
            # return h(t), y(t)
            h_t = T.switch(
                T.eq(is_start, 1),
                self.f(xw_t + h0.dot(self.Wh) + self.bh),
                self.f(xw_t + h_t1.dot(self.Wh) + self.bh)
            )
            return h_t

        h, _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0],
            sequences=[XW, thStartPoints],
            non_sequences=[self.h0],
            n_steps=XW.shape[0],
            # mode="DebugMode"
        )

        # py_x = y[:, 0, :]
        py_x = T.nnet.softmax(h.dot(self.Wo) + self.bo)
        prediction = T.argmax(py_x, axis=1)

        ## Notes
        # py_x[T.arange(thY.shape[0]), thY] ==> is advence indexing
        # eg:
        #   thY = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        #   py_x = array([
        #           [0.21464644, 0.78535356], [0.53838035, 0.46161965], [0.53524101, 0.46475899], [0.4911481 , 0.5088519 ],
        #           [0.49989071, 0.50010929], [0.53311029, 0.46688971], [0.49294333, 0.50705667], [0.49984173, 0.50015827],
        #           [0.49985361, 0.50014639], [0.49982706, 0.50017294], [0.53299261, 0.46700739], [0.49291816, 0.50708184]
        #         ])
        #  py_x[T.arange(thY.shape[0]), thY] ==> py_x[[0,1,2,3,4,5,6,7,8,9,10,11], [1,1,1,0,1,1,0,1,0,1,1,0]
        #                                    ==> [ 0.78535356, 0.46161965, 0.46475899, 0.4911481 ,
        #                                          0.50010929, 0.46688971, 0.49294333, 0.50015827,
        #                                          0.49985361, 0.50017294, 0.46700739, 0.49291816]
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        # self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY, thStartPoints],
            outputs=[cost, prediction, py_x],
            updates=updates
            # mode="DebugMode"
        )

        costs = []
        n_batches = N // batch_sz
        sequenceLength = X.shape[1]

        startPoints = np.zeros(sequenceLength*batch_sz, dtype=np.int32)
        for b in range(batch_sz):
            startPoints[b*sequenceLength] = 1
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j+1)*batch_sz].reshape(sequenceLength*batch_sz, D)
                Ybatch = Y[j*batch_sz:(j+1)*batch_sz].reshape(sequenceLength*batch_sz).astype(np.int32)
                c, p, rout = self.train_op(Xbatch, Ybatch, startPoints)
                cost += c

                for b in range(batch_sz):
                    idx = sequenceLength*(b + 1) - 1
                    if p[idx] == Ybatch[idx]:
                        n_correct += 1
            print("shape y:", rout.shape)
            print("i:", i, "cost:", cost, "Classification rate:", (float(n_correct) / N))
            costs.append(cost)
            # if n_correct == N:
            #     break
        if show_fig:
            plt.plot(costs)
            plt.show()


def parity(B=12, learning_rate=1e-3, epochs=100):
    X, Y = all_parity_pairs_with_sequence_labels(B)

    rnn = SimpleRNN(4)
    rnn.fit(X, Y,
        batch_sz=10,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=T.nnet.sigmoid,
        show_fig=False
    )


if __name__ == '__main__':
    parity()