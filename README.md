# sentence2vec
一个将句子转化为向量表征的工具库，并集成一些常用的算法。参考sklearn库的用法，尽可能地做到简单使用，后续会持续更新。

输入：句子组成的list，如：['I like natural language processing', ..., 'This is an example']

输出：[[0.1, 0.1, ..., 0.1], ..., [0.1, 0.1, ..., 0.1]]



## 依赖
- python 3.6
- numpy 1.17.0
- gensim 3.6.0
- scikit-learn 0.21.2

上述版本号仅供参考。



## 当前实现

Model | Year | Status 
:-: | :-: | :-: 
SIF<sup>[[1](#reference1)]</sup> (smooth inverse frequency) | 2016 | Finished 
CPM<sup>[[2](#reference2)]</sup> (concatenated power mean) | 2018 | Plan 



## 实例

见example.py

example_sif.py:

```python
from sentence2vec.utils import glove2w2v
from sentence2vec.SIF import SIF

######## 转换向量格式 ########
# 由于使用gensim的api进行转换，因此请填写绝对路径
glove_file = 'C:/data/glove.840B.300d.txt'    # download from https://nlp.stanford.edu/projects/glove/
w2v_file = 'C:/data/glove_w2v.840B.300d.txt'
glove2w2v(glove_file, w2v_file)
################################

sentences = ['I like natural language processing', 'This is an example']   # 所有句子list
weight_file = './data/weight_file.txt'   # 权重存储路径
weight_para = 1e-3   # 参考论文
rmpc = 1   # 参考论文

sif = SIF(sentences, w2v_file, weight_file, weight_para, rmpc)
sentences_embedding = sif.transform()
print(len(sentences_embedding), len(sentences_embedding[0]))
```



## Reference

<div id='reference1'></div>

[1] Arora S, Liang Y, Ma T. A simple but tough-to-beat baseline for sentence embeddings[J]. 2016.

<div id='reference2'></div>

[2] Rücklé A, Eger S, Peyrard M, et al. Concatenated power mean word embeddings as universal cross-lingual sentence representations[J]. arXiv preprint arXiv:1803.01400, 2018.



## To-Do

- [ ] pip install
- [ ] more models



## Other

- 参考的github请见github_list.txt；
- 发现bug或者问题请提issue，谢谢。