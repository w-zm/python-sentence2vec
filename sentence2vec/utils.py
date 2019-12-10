from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

def glove2w2v(glove_path, w2v_path):
    glove_path = datapath(glove_path)
    w2v_path = get_tmpfile(w2v_path)
    glove2word2vec(glove_path, w2v_path)

