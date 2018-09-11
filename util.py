import numpy as np, sys, string

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

def all_parity_pairs_with_sequence_labels(nbit):
    X, Y = all_parity_pairs(nbit)
    N, t = X.shape

    # we want every time step to have a label
    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n,i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n,i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)
    return X, Y_t

# unfortunately Python 2 and 3 translates work differently
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def get_robert_frost():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in open('/home/zero/RNN_Lecture_codes/machine_learning_examples/hmm_class/robert_frost.txt'):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx