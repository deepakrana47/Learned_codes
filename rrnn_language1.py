import theano, theano.tensor as T, numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost

class SimpleRNN:
    def __init__(self, D, M, V):
        self.D = D
        self.M = M
        self.V = V

    def fit(self, X, learning_rate=10e-1, mu=.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V

        # initial weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        # z = np.ones(M)
        Wxz = init_weight(D, M)
        Whz = init_weight(M, M)
        bhz = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)
        h0 = np.zeros(M)

        thX, thY, py_x, prediction = self.set(We, Wx, Wh, bh, Wxz, Whz, bhz, Wo, bo, h0, activation)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]

        updates = [
                      (p, p + mu * dp - learning_rate * g) for p, dp, g in zip(self.params, dparams, grads)
                  ] + [
                      (dp, mu * dp - learning_rate * g) for dp, g in zip(dparams, grads)
                  ]

        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
        )

        costs = []
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                if np.random.random() < .1:
                    input_seq = [0] + X[j]
                    output_Seq = X[j] + [1]
                else:
                    input_seq = [0] + X[j][:-1]
                    output_Seq = X[j]

                n_total += len(output_Seq)
                c, p= self.train_op(input_seq, output_Seq)
                cost += c
                for pj, xj in zip(p, output_Seq):
                    if pj == xj:
                        n_correct += 1
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct) / n_total))
            costs.append(cost)
            # if n_correct == N:
            #     break

        if show_fig:
            plt.plot(costs)
            plt.show()

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename, activation):
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        # z = npz['arr_4']
        Wxz = npz['arr_4']
        Whz = npz['arr_5']
        bhz = npz['arr_6']
        Wo = npz['arr_7']
        bo = npz['arr_8']
        h0 = npz['arr_9']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, Wxz, Whz, bhz, Wo, bo, h0, activation)
        return rnn

    def set(self, We, Wx, Wh, bh, Wxz, Whz, bhz, Wo, bo, h0, activation):
        self.f = activation
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bhz = theano.shared(bhz)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.h0 = theano.shared(h0)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.Wxz,self.Whz,self.bhz, self.Wo, self.bo, self.h0]

        thX = T.ivector('X')
        Ei = self.We[thX] # TxD
        thY = T.ivector('Y')

        def recurrent(x_t, h_t1):
            # return h(t), y(t)
            hhat_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + hhat_t.dot(self.Whz) + self.bhz)
            h_t = (1 - z_t) * h_t1 + z_t * hhat_t
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrent,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True,
        )
        return thX, thY, py_x, prediction

    def generate(self, pi, word2idx):
        idx2word = {v:k for k,v in word2idx.items()}
        V = len(pi)

        n_lines = 0

        X = [np.random.choice(V, p=pi) ]
        print(idx2word[X[0]], end=' ')

        while n_lines < 4:
            P = self.predict_op(X)[-1]
            X += [P]

            if P > 1:
                word = idx2word[P]
                print(word, end=' ')
            elif P == 1:
                n_lines += 1
                print('')
                if n_lines < 4:
                    X = [np.random.choice(V, p=pi)]
                    print(idx2word[X[0]], end=' ')

def train_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(30, 30, len(word2idx))
    rnn.fit(sentences, learning_rate=10e-5, show_fig=True, epochs=1000, activation=T.nnet.relu)
    rnn.save("RRNN1_D30_M30_epochs1000_relu.npz")

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load("RRNN1_D30_M30_epochs1000_relu.npz", activation=T.nnet.relu)

    V = len(word2idx)
    pi = np.zeros(V)
    for sentence in sentences:
        pi[sentence[0]] += 1
    pi/=pi.sum()

    rnn.generate(pi, word2idx)

if __name__ == '__main__':
    train_poetry()
    # generate_poetry()