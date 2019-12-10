"""smooth inverse frequency (SIF).
"""
from gensim.models import KeyedVectors

class SIF():
    def __init__(self, word_file, weight_file, weight_para, rmpc=1):
        self.word_file = word_file
        self.weight_file = weight_file
        self.weight_para = weight_para
        self.rmpc = rmpc

        print('--- load word vectors ---')
        (words, We) = self.getWordmap(word_file)
        print('--- finish load word vectors ---')
        print('--- load word weights ---')
        word2weight = self.getWordWeight(self.weight_file, self.weight_para)  # word2weight['str'] is the weight for the word 'str'
        print('--- finish load word weights ---')
        print('--- start weight4ind ---')
        weight4ind = getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
        print('--- finish weight4ind ---')

    def getWordmap(self, word_file):
        w2v_model = KeyedVectors.load_word2vec_format(word_file)
        word2ids = w2v_model.wv.vocab
        words = {}
        for word, vocab in word2ids.items():
            words[word] = vocab.index
        return (words, w2v_model.vectors)

    def getWordWeight(weightfile, a=1e-3):
        if a <= 0:  # when the parameter makes no sense, use unweighted
            a = 1.0

        word2weight = {}
        with open(weightfile, encoding='utf-8') as f:
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
                    print(i)
            else:
                print('aaaaaaaaa')
                print(i)
                print('aaaaaaaaa')
        for key, value in word2weight.items():
            word2weight[key] = a / (a + value / N)
        return word2weight