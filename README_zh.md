# sentence2vec
一个将句子转化为向量表征的工具库，并集成一些常用的算法。参考sklearn库的用法，尽可能地做到简单使用，后续会持续更新。

输入：句子组成的list，如：['I like natural language processing', ..., 'This is an example']

输出：[[0.1, 0.1, ..., 0.1], ..., [0.1, 0.1, ..., 0.1]]

## 依赖
- python 3.6

- gensim 3.6.0

- scikit-learn 0.21.2

上述版本号仅为参考。

## 当前实现

Model | Year | Status 
:-: | :-: | :-: 
SIF<sup>[1]</sup> (smooth inverse frequency) | 2016 | in progress 
CPM<sup>[2]</sup> (concatenated power mean) | 2018 | plan 

## 实例





## Reference

[1] Arora S, Liang Y, Ma T. A simple but tough-to-beat baseline for sentence embeddings[J]. 2016.

[2] Rücklé A, Eger S, Peyrard M, et al. Concatenated power mean word embeddings as universal cross-lingual sentence representations[J]. arXiv preprint arXiv:1803.01400, 2018.

