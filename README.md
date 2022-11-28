# 数据集
FB15K

# TransE的numpy实现和torch实现

本仓库为个人学习所用，借鉴了

https://github.com/Anery/transE

https://github.com/zqhead/TransE

因为代码比较老，相对更适合新手，建议了解transe的基本思想和代码以后，不必深究本代码每一处的实现细节，后续可以仔细研究其他更先进的kge方法

例如Rotate

https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

Conve

https://github.com/TimDettmers/ConvE

## 感谢原代码作者的贡献，本代码整体易于理解，适合初学者学习，我在代码中加入了一些自己理解的详细注释，现在已经更新pytorch版本
	
### 文件结构说明
1. 训练和测试的代码放在src文件夹下
2. 训练和测试的结果放在res文件夹下，经过1001个epoch的训练，损失约为14000（其实300个epoch的时候就基本固定了）。

###  运行
运行时只需要调整输入和保存文件夹的位置，直接运行transe_simple.py即可

#### 关于transE：
论文原文：[Translating embeddings for modeling multi-relational data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela)

#### 1 训练数据

FB15k.

#### 2. 伪代码

![image](https://user-images.githubusercontent.com/68625084/166636446-ee7ae1dc-778a-4270-96f6-679868e6d420.png)

伪代码的意思是：

input: 输入模型的参数是训练集的三元组，实体集E，关系集L，margin，向量的维度k

1：初始化： 对于关系按照1的初始化方式初始化即可

2：这里进行了L2范数归一化，也就是除以自身的L2范数

3：同理，也对实体进行了初始化，但是这里没有除以自身的L2范数

4：训练的循环过程中：

5：首先对实体进行了L2范数归一化

6：取一个batch的样本，这里Sbatch代表的是正样本，也就是正确的三元组

7： 初始化三元组对，应该就是创造一个用于储存的列表

8，9，10：这里的意思应该是根据Sbatch的正样本替换头实体或者尾实体构造负样本，然后把对应的正样本三元组和负样本三元组放到一起，组成Tbatch

11：完成正负样本的提取

12：根据梯度下降更新向量

13：结束循环

#### 4. 需要注意的点

详细见知乎 https://zhuanlan.zhihu.com/p/508508180?

 #### 5. 测试
 
- isFit参数：区分raw和filter。filter会非常慢。

#### 6. 结果

##### 针对FB15k

训练1000个epochs的loss：因为是使用了累加的loss，所以看起来比较大，最后效果还不错

epoch: 900  loss: 14122.820245424562

epoch: 910 loss: 14373.68032895213

epoch: 920 loss: 14340.662277325615

epoch: 930 loss: 14373.677382376287

epoch: 940 loss: 14328.833943474272

epoch: 950 loss: 14310.58852751293

epoch: 960 loss: 14262.76358291793

epoch: 970 loss: 14311.827534107646

epoch: 980 loss: 14327.824546415322

epoch: 990 loss: 14146.539213775186

现在已经修改为了每个batch的平均loss，但是没有再跑一遍，效果是一样的。



##### 测试结果：
entity hits@10: 0.3076551945963332

entity meanrank: 254.52704372704034

relation hits@10: 0.7906586988539216

relation meanrank: 81.79988488429179


#### 更新了torch版本及详细注释

torch的训练集、验证集、测试集作者均做了一些修改，其实就是把尾实体和关系的位置调换了一下，保持（头实体、关系、尾实体）这样的位置排布

如果需要这部分修改后的数据可以在https://github.com/zqhead/TransE  下载

基本思想和numpy版本的是相同的，但是实现同一个功能的方式有些区别，主要使用了一些torch自己的工具，因为用了gpu，相对而言，训练速度很快，供大家参考。

下面介绍一些我理解的主要需要注意的地方：

1. 创造负样本（错误三元组）的时候，这里使计算了平均尾节点数 hpt 和平均头结点数tphtph 表示每一个头结对应的平均尾节点数 hpt 表示每一个尾节点对应的平均头结点数
当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体

举例说明 ：
在一个知识图谱中，一共有10个实体 和n个关系，如果其中一个关系使两个头实体对应五个尾实体，那么这些头实体的平均 tph为2.5，而这些尾实体的平均 hpt只有0.4，则此时我们更倾向于替换头实体，
因为替换头实体才会有更高概率获得正假三元组，如果替换头实体，获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9

2. 初始化向量的方式借助了torch的工具

（1）首先使用nn.Embedding对实体和关系向量化

（2）使用Xavier初始化替代原文的初始化方法（深度神经网络中的 Xavier 初始化）

（3）然后再进行L2范数归一化

3. 将实体向量和关系向量视为模型优化的参数，使用SGD或者Adam优化器进行优化



测试结果：

hits@10: 0.4067393475647949

meanrank: 246.31837111272876
