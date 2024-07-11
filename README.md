## If you want to use your own data to train TransE, see the TransE-mydataset.rar file

Because the code is relatively old and more suitable for beginners, it is recommended that after understanding the basic ideas and code of transE, there is no need to delve into every implementation detail of this code. In the future, other more advanced kge methods can be carefully studied

For example, Rotate

https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

Conve

https://github.com/TimDettmers/ConvE

SelectE

https://github.com/zulihit/SelectE
	
### Organization
1. The code for training and testing is located in the src folder
2. The results of training and testing are in the res folder. After 1001 epochs of training, the loss is about 14000 (in fact, it is basically fixed at 300 epochs).

###  To reproduce the results
Just adjust the location of the DATA and save folders, run transe_simplic.py directly

#### TransE：
Paper：[Translating embeddings for modeling multi-relational data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela)

#### 1 Train data

FB15k.

#### 2. Pseudo code

![image](https://user-images.githubusercontent.com/68625084/166636446-ee7ae1dc-778a-4270-96f6-679868e6d420.png)

The meaning of pseudocode is:

Input: The parameters of the input model are the triplet of the training set, entity set E, relationship set L, margin, and vector dimension k

1: Initialization: Initialize the relationship according to the initialization method of 1

2: L2 norm normalization has been performed here, which means dividing by its own L2 norm

3: Similarly, the entity has also been initialized, but here it is not divided by its own L2 norm

4: During the training cycle:

5: Firstly, L2 norm normalization was performed on the entity

6: Take a batch of samples, where Sbatch represents the positive sample, which is the correct triplet

7: Initialize triplet pairs by creating a list for storage

8, 9, 10: The meaning here should be to replace the head or tail entity of the Sbatch with positive samples to construct negative samples, and then put the corresponding positive and negative sample triplets together to form Tbatch

11: Complete the extraction of positive and negative samples

12: Update vectors based on gradient descent

13: End cycle

#### 4. Key points

ZHIHU https://zhuanlan.zhihu.com/p/508508180?

 #### 5. Test
 
- isFit：Distinguish between raw and filter. The filter will be very slow.

#### 6. Results

##### For FB15k

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


##### Results：
entity hits@10: 0.3076551945963332

entity meanrank: 254.52704372704034

relation hits@10: 0.7906586988539216

relation meanrank: 81.79988488429179

# Acknowledgement

This repo benefits from these repos. Thanks for their wonderful works.

https://github.com/Anery/transE

https://github.com/zqhead/TransE



Final results：

hits@10: 0.4067393475647949

meanrank: 246.31837111272876
