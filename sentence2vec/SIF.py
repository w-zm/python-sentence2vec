"""smooth inverse frequency (SIF).
"""
from gensim.models import KeyedVectors

class SIF():
    def __init__(self, word_file, weight_file, weight_para, data, rmpc=1):
        self.word_file = word_file
        self.weight_file = weight_file
        self.weight_para = weight_para
        self.data = data
        self.rmpc = rmpc

        print('--- load word vectors ---')
        (words, We) = self.getWordmap(word_file)
        print('--- finish load word vectors ---')
        print('--- load word weights ---')
        word2weight = self.getWordWeight(self.weight_file, self.weight_para)  # word2weight['str'] is the weight for the word 'str'
        print('--- finish load word weights ---')
        print('--- start weight4ind ---')
        weight4ind = self.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
        print('--- finish weight4ind ---')

    # def transform(self):
    #     x, m = self.sentences2idx(self.data, words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    #     w = seq2weight(x, m, weight4ind)  # get word weights
    #     embedding = SIF_embedding(We, x, w, params)

    def getWordmap(self, word_file):
        w2v_model = KeyedVectors.load_word2vec_format(word_file)
        word2ids = w2v_model.wv.vocab
        words = {}
        for word, vocab in word2ids.items():
            words[word] = vocab.index
        return (words, w2v_model.vectors)

    def getWordWeight(self, weight_file, a=1e-3):
        if a <= 0:  # when the parameter makes no sense, use unweighted
            a = 1.0

        word2weight = {}
        with open(weight_file, encoding='utf-8') as f:
            lines = f.readlines()
        N = 0
        for i in lines:
            i = i.strip()
            if (len(i) > 0):
                i = i.split()
                if (len(i) == 2):
                    word2weight[i[0]] = float(i[1])
                    N += float(i[1])
                else:
                    print('format error')
                    exit()
            else:
                print('format error')
                exit()
        for key, value in word2weight.items():
            word2weight[key] = a / (a + value / N)
        return word2weight

    def getWeight(self, words, word2weight):
        weight4ind = {}
        for word, ind in words.items():
            if word in word2weight:
                weight4ind[ind] = word2weight[word]
            else:
                weight4ind[ind] = 1.0
        return weight4ind

    def sentences2idx(self, sentences, words):
        seq1 = []
        for i in sentences:
            seq1.append(self.getSeq(i, words))
        x1, m1 = self.prepare_data(seq1)
        return x1, m1

    def getSeq(self, p1, words):
        p1 = p1.split()
        X1 = []
        for i in p1:
            X1.append(self.lookupIDX(words, i))
        return X1

    def lookupIDX(self, words, w):
        return words.get(w, len(words) - 1)

    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype('float32')
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype='float32')
        return x, x_mask

    def seq2weight(self, seq, mask, weight4ind):
        weight4ind_np = np.zeros((len(weight4ind),))
        for index, weight in weight4ind.items():
            weight4ind_np[index] = weight

        weight = np.zeros(seq.shape).astype('float32')
        for index, row in enumerate(seq):
            weight[index] = weight4ind_np[row]
        weight = weight * mask
        return weight