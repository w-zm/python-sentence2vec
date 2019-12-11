"""smooth inverse frequency (SIF).
"""
import numpy as np

from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD

class Params(object):
    def __init__(self):
        self.LW = 1e-5
        self.LC = 1e-5
        self.eta = 0.05

    def __str__(self):
        t = "LW", self.LW, ", LC", self.LC, ", eta", self.eta
        t = list(map(str, t))
        return ' '.join(t)

class SIF(object):
    def __init__(self, word_file, weight_file, sentences, weight_para=1e-3, rmpc=1):
        self.word_file = word_file
        self.weight_file = weight_file
        self.sentences = sentences
        self.weight_para = weight_para if weight_para > 0 else 1.0     # when the parameter makes no sense, use unweighted
        self.rmpc = rmpc

        print('--- load word vectors ---')
        self.words, self.We = self.getWordmap()
        print('--- finish load word vectors ---')
        print('--- load word weights ---')
        self.word2weight = self.getWordWeight()  # word2weight['str'] is the weight for the word 'str'
        print('--- finish load word weights ---')
        print('--- start weight4ind ---')
        self.weight4ind = self.getWeight()  # weight4ind[i] is the weight for the i-th word
        print('--- finish weight4ind ---')

    def transform(self):
        params = Params()
        params.rmpc = self.rmpc

        x, m = self.sentences2idx(self.sentences, words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = self.seq2weight(x, m)  # get word weights
        embedding = self.SIF_embedding(x, w, params)

        return embedding

    def getWordmap(self):
        w2v_model = KeyedVectors.load_word2vec_format(self.word_file)
        word2ids = w2v_model.wv.vocab
        words = {}
        for word, vocab in word2ids.items():
            words[word] = vocab.index
        return words, w2v_model.vectors

    def getWordWeight(self):
        word2weight = {}
        with open(self.weight_file, encoding='utf-8') as f:
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
            word2weight[key] = self.weight_para / (self.weight_para + value / N)
        return word2weight

    def getWeight(self):
        weight4ind = {}
        for word, ind in self.words.items():
            if word in self.word2weight:
                weight4ind[ind] = self.word2weight[word]
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

    def seq2weight(self, seq, mask):
        weight4ind_np = np.zeros((len(self.weight4ind),))
        for index, weight in self.weight4ind.items():
            weight4ind_np[index] = weight

        weight = np.zeros(seq.shape).astype('float32')
        for index, row in enumerate(seq):
            weight[index] = weight4ind_np[row]
        weight = weight * mask
        return weight

    def SIF_embedding(self, x, w, params):
        """
        Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in the i-th sentence
        :param w: w[i, :] are the weights for the words in the i-th sentence
        :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
        :return: emb, emb[i, :] is the embedding for sentence i
        """
        emb = self.get_weighted_average(x, w)
        if params.rmpc > 0:
            emb = self.remove_pc(emb, params.rmpc)
        return emb

    def get_weighted_average(self, x, w):
        """
        Compute the weighted average vectors
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in sentence i
        :param w: w[i, :] are the weights for the words in sentence i
        :return: emb[i, :] are the weighted average vector for sentence i
        """
        n_samples = x.shape[0]
        emb = np.zeros((n_samples, self.We.shape[1]))
        for i in range(n_samples):
            emb[i, :] = w[i, :].dot(self.We[x[i, :], :]) / np.count_nonzero(w[i, :])
        return emb

    def remove_pc(self, X, npc=1):
        """
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(X, npc)
        if npc == 1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX

    def compute_pc(self, X, npc=1):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_