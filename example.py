from sentence2vec.utils import glove2w2v

glove_file = './data/glove.840B.300d.txt'
w2v_file = './data/glove_w2v.840B.300d.txt'

glove2w2v(glove_file, w2v_file)
