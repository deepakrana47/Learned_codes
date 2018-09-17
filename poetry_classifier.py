import theano, theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import init_weight, get_poetry_classifier_data

class SimpleRNN:
    def __init__(self, M, V):
        self.M = M
        self.V = V

    def set(self, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thx = T.ivector('X')
        thy = T.iscalar('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(self.Wx[x_t] + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            sequences=thx,
            outputs_info=[self.h0, None],
            n_steps=thx.shape[0],
        )
        py = y[-1, 0, :]
        prediction = T.argmax(py)
        self.predict_op = theano.function(
            inputs=[thx],
            outputs=prediction,
            allow_input_downcast=True
        )
        return thx, thy, py, prediction

    def fit(self, X, Y, learning_rate=10e-1, mu=.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):

        M = self.M
        V = self.V
        k = len(set(Y))

        X, Y = shuffle(X, Y)
        Nvalid = 10
        Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:]
        X, Y = X[:-Nvalid], Y[:-Nvalid]

        Wx = init_weight(V, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, k)
        bo = np.zeros(k)

        ## setting tensor variables
        # self.f = activation
        # self.Wx = theano.shared(Wx)
        # self.Wh = theano.shared(Wh)
        # self.bh = theano.shared(bh)
        # self.h0 = theano.shared(h0)
        # self.Wo = theano.shared(Wo)
        # self.bo = theano.shared(bo)
        # self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]
        #
        # thx = T.ivector('X')
        # thy = T.iscalar('Y')
        #
        # def recurrence(x_t, h_t1):
        #     # returns h(t), y(t)
        #     h_t = self.f(self.Wx[x_t] + h_t1.dot(self.Wh) + self.bh)
        #     y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
        #     return h_t, y_t
        #
        # [h, y], _ = theano.scan(
        #     fn = recurrence,
        #     sequences = thx,
        #     outputs_info = [self.h0, None],
        #     n_steps = thx.shape[0],
        # )
        # py = y[-1, 0, :]
        # prediction = T.argmax(py)
        # self.predict_op = theano.function(
        #     inputs=[thx],
        #     outputs=prediction,
        #     allow_input_downcast=True
        # )
        ## end setting tensor variables
        thx, thy, py_x, prediction = self.set(Wx, Wh, bh, h0, Wo, bo, activation)

        cost = -T.mean(T.log(py_x[thy]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        lr = T.scalar('learning rate')
        updates =   [
                        (p, p - lr*g + mu*dp) for p, g, dp in zip(self.params, grads, dparams)
                    ] + [
                        (dp, mu*dp - lr*g) for g, dp in zip(grads, dparams)
                    ]

        self.train_op = theano.function(
            inputs = [thx, thy, lr],
            outputs = [cost, prediction],
            updates = updates,
            allow_input_downcast=True,
        )

        costs = []
        n_total = len(X)
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(n_total):
                c, py = self.train_op(X[j], Y[j], learning_rate)
                # if np.isnan(c):
                #     print
                cost += c
                if py == Y[j]:
                    n_correct += 1
            learning_rate *= .9999

            n_correct_valid = 0
            for j in range(Nvalid):
                p = self.predict_op(Xvalid[j])
                if p == Yvalid[j]:
                    n_correct_valid += 1

            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct) / n_total))
            print("validation correct rate:",(float(n_correct_valid/Nvalid)))
            costs.append(cost)


        if show_fig:
            plt.plot(costs)
            plt.show()

def train_poetry():
    X, Y, V = get_poetry_classifier_data(sample_per_class=500)
    rnn = SimpleRNN(30, V)
    rnn.fit(X, Y, learning_rate=10e-6, show_fig=True, activation=T.nnet.relu, epochs=500)

if __name__ == '__main__':
    # a = get_poetry_classifier_data(sample_per_class=10)
    train_poetry()