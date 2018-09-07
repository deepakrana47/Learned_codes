import numpy as np

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo)/np.sqrt(Mi + Mo)

def all_parity_pairs(nbit):
    N = 2**nbit

    # for making multiple of 100
    remainder = 100 - (N % 100)
    Ntotal = N + remainder

    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N

        for j in range(nbit):
            if i%(2**(j+1)) != 0:
                i -= 2**j
                X[ii, j] = 1
        Y[ii] = X[ii].sum() % 2
    return  X, Y
