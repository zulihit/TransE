# transE
## 以下为我的贡献
### 加速training
原作者的代码写得很清晰，读起来很舒服，但缺点就是跑起来太慢了。<br/>
在参数设置为`embedding_dim=50, learning_rate=0.01, margin=1, L1=True，batch_num=400`（原代码中的设置）的情况下，在我的PC上跑一个epoch需要约180s的时间。<br/>
经过我的修改，在不改变原代码逻辑的前提下，将一个epoch的训练时间缩短到30s。

### 加速testing
对测试过程的代码也做了一点修改，大概提高了3倍的速度。
在不改变原代码的逻辑前提下，将每个测试的triple所需的时间由0.75s降到0.25s（更换头和尾后排序），由0.3s降到0.1s（更换relation后排序）。
	
### 文件结构说明
1. 训练和测试的代码放在src文件夹下
2. 训练和测试的结果放在res文件夹下，经过1001个epoch的训练，损失约为14000（其实300个epoch的时候就基本固定了）。

---
## 以下为原作者的贡献
#### 关于transE：

1、论文原文：[Translating embeddings for modeling multi-relational data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela)

2、[我的一篇笔记](https://blog.csdn.net/shunaoxi2313/article/details/89766467)

#### 1 训练数据

FB15k. 

其它数据（如WorldNet等），见(https://github.com/thunlp/KB2E)

#### 2. 训练transE

- Tbatch更新：在update_embeddings函数中有一个deepcopy操作，目的就是为了批量更新。这是ML中mini-batch SGD的一个通用的训练知识，在实际编码时很容易忽略。
- 两次更新：update_embeddings函数中，要对correct triplet和corrupted triplet都进行更新。虽然写作$(h,l,t)$和$(h',l,t')$，但两个三元组只有一个entity不同（前面说了，不同时替换头尾实体），所以在每步更新时重叠的实体要更新两次（和更新relation一样），否则就会导致后一次更新覆盖前一次。
- 关于L1范数的求导方法：参考了[刘知远组实现](https://github.com/thunlp/KB2E)中的实现，是先对L2范数求导，逐元素判断正负，为正赋值为1，负则为-1。
- 超参选择：对FB15k数据集，epoch选了1000（其实不需要这么大，后面就没什么提高了），nbatches选了400（训练最快），embedding_dim=50, learning_rate=0.01, margin=1。


 #### 3. 测试
- isFit参数：区分raw和filter。filter会非常慢。
