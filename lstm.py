import sys
import theano, theano.tensor as T
import numpy as np

from util import init_weight

class LSTM:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f = activation

        Wxi = init_weight(Mi, Mo)
        Whi = init_weight(Mo, Mo)
        Wci = init_weight(Mo, Mo)
        bi = np.zeros(Mo)
        Wxf = init_weight(Mi, Mo)
        Whf = init_weight(Mo, Mo)
        Wcf = init_weight(Mo, Mo)
        bf = np.zeros(Mo)
        Wxo = init_weight(Mi, Mo)
        Who = init_weight(Mo, Mo)
        Wco = init_weight(Mo, Mo)
        bo = np.zeros(Mo)
        Wxc = init_weight(Mi, Mo)
        Whc = init_weight(Mo, Mo)
        bc = np.zeros(Mo)
        h0 = np.zeros(Mo)
        c0 = np.zeros(Mo)

        self.Wxi = theano.shared(Wxi)
        self.Whi = theano.shared(Whi)
        self.Wci = theano.shared(Wci)
        self.bi = theano.shared(bi)
        self.Wxf = theano.shared(Wxf)
        self.Whf = theano.shared(Whf)
        self.Wcf = theano.shared(Wcf)
        self.bf = theano.shared(bf)
        self.Wxo = theano.shared(Wxo)
        self.Who = theano.shared(Who)
        self.Wco = theano.shared(Wco)
        self.bo = theano.shared(bo)
        self.Wxc = theano.shared(Wxc)
        self.Whc = theano.shared(Whc)
        self.bc = theano.shared(bc)
        self.h0 = theano.shared(h0)
        self.c0 = theano.shared(c0)
        self.params = [self.Wxi, self.Whi, self.Wci, self.bi, self.Wxf, self.Whf, self.Wcf, self.bf, self.Wxo, self.Who, self.Wco, self.bo, self.Wxc, self.Whc, self.bc, self.h0, self.c0]

    def recurrence(self, x_t, h_t1, c_t1):
        i = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        f = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        chat_t = T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        c_t = f * c_t1 + i * chat_t
        o = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o * T.tanh(c_t)
        return h_t, c_t

    def output(self, X):
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=X,
            outputs_info=[self.h0, self.c0],
            n_steps=X.shape[0],
        )
        return h