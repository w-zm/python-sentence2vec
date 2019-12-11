from sentence2vec.utils import glove2w2v
from sentence2vec.SIF import SIF

######## 转换向量格式 ########
# 由于使用gensim的api进行转换，因此请填写绝对路径
glove_file = 'C:/work/learn/sentence2vec/data/glove.840B.300d.txt'    # download from https://nlp.stanford.edu/projects/glove/
w2v_file = 'C:/work/learn/sentence2vec/data/glove_w2v.840B.300d.txt'
glove2w2v(glove_file, w2v_file)
################################

sentences = ['I like natural language processing', 'This is an example']   # 所有句子list
weight_file = './data/weight_file.txt'   # 权重存储路径
weight_para = 1e-3   # 参考论文
rmpc = 1   # 参考论文

sif = SIF(sentences, w2v_file, weight_file, weight_para, rmpc)
sentences_embedding = sif.transform()
print(len(sentences_embedding), len(sentences_embedding[0]))