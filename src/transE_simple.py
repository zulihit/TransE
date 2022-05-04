import codecs
import random
import math
import numpy as np
import copy
import time

entity2id = {}
relation2id = {}


def data_loader(file):
    file1 = file + "train.txt"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with codecs.open(file2, 'r') as f1, codecs.open(file3, 'r') as f2:  # entity2id和relation2id
        lines1 = f1.readlines()  # readlines：作为列表返回文件中的所有行，其中每一行都是列表对象中的一项
        lines2 = f2.readlines()  # 也就是将每一行拿出来组成一个列表返回
        for line in lines1:
            line = line.strip().split('\t')  # 这一步是拆分出实体和编号    strip()：删除头尾的空格     \t：水平制表符相当于tab
            if len(line) != 2:  # 如果这一行不是两个元素，就放弃这一行
                continue
            entity2id[line[0]] = int(line[1])  # 转换成字典{实体：id}

        for line in lines2:  # 同理，制作关系和编号的字典
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = int(line[1])

    entity_set = set()  # 用set去重
    relation_set = set()
    triple_list = []

    with codecs.open(file1, 'r') as f:  # 文件读尽量用codecs.open方法，一般不会出现编码的问题
        content = f.readlines()  # 读取训练集
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = entity2id[triple[0]]  # 找到训练集中三元组的对应编号
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            triple_list.append([h_, t_, r_])  # 储存三元组的编号

            entity_set.add(h_)  # 储存头实体的编号
            entity_set.add(t_)  # 储存尾实体的编号

            relation_set.add(r_)  # 储存关系的编号

    return entity_set, relation_set, triple_list  # 返回实体的编号集合，关系的编号集合，三元组的编号集合


def distanceL2(h, r, t):
    # 为方便求梯度，去掉sqrt（sqrt是计算平方根的函数）
    return np.sum(np.square(h + r - t))


def distanceL1(h, r, t):
    return np.sum(np.fabs(h + r - t))  # fabs()方法返回数字的绝对值


class TransE:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1 = L1

        self.loss = 0

    def emb_initialize(self):
        relation_dict = {}
        entity_dict = {}

        for relation in self.relation:
            # 初始化关系
            # np.random.uniform从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),  # math.sqrt正平方根
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            # 伪代码中关系向量初始化的办法（L2范数归一化）：关系向量除以它的L2范数，每个r_emb_temp代表一个关系的向量
            # 关于L2范数归一化：https://www.pianshen.com/article/9664381455/
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)  # np.linalg.norm求范数，ord=2代表二范数

        for entity in self.entity:
            # 初始化实体
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        self.relation = relation_dict  # 初始化后的关系  {关系编号：L2范数归一化后的向量}
        self.entity = entity_dict  # 初始化后的实体   {实体编号：L2范数归一化后的向量}

    def train(self, epochs):
        nbatches = 400  # 根据batchsize计算的，看有多少个batch
        batch_size = len(self.triple_list) // nbatches  # 根据三元组数量来判断batch_size
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0
            num = 0
            for k in range(nbatches):
                # Sbatch:list
                # start1 = time.time()
                # random.sample() 返回一个列表，其中包含从序列中随机选择的指定数量的项目,此处返回一个随机取batchsize数量的样本
                Sbatch = random.sample(self.triple_list, batch_size)  # 取一个batch的三元组样本
                Tbatch = []  # 负样本
                num = num + 1
                # 每个triple选3个负样例
                # for i in range(3):
                for triple in Sbatch:  # 创造负样本
                    corrupted_triple = self.Corrupt(triple)
                    Tbatch.append((triple, corrupted_triple))  # 包含正样本和负样本的列表

                self.update_embeddings(Tbatch)  # 更新向量
                self.loss = self.loss / num
                # end1 = time.time()
                # print('time of one batch: %s'%(round((end1 - start1),3)))
                # return

            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))  # 打印
            print("loss: ", self.loss)

            # 保存每一个batch的临时结果
            if epoch % 10 == 0:
                with codecs.open("../res/entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(str(e) + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("../res/relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        f_r.write(str(r) + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")
                with codecs.open("../res/result_temp", "a") as f_s:
                    f_s.write("epoch: %d\tloss: %s\n" % (epoch, self.loss))

        print("写入文件...")
        with codecs.open("../res/entity_50dim_batch400", "w") as f1:
            for e in self.entity.keys():
                f1.write(str(e) + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("../res/relation_50dim_batch400", "w") as f2:
            for r in self.relation.keys():
                f2.write(str(r) + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        print("写入完成")

    def Corrupt(self, triple):  # 随机替换头实体和尾实体制作负样本
        corrupted_triple = copy.deepcopy(triple)  # deepcopy：将被复制对象完全再复制一遍作为独立的新个体单独存在
        seed = random.random()
        if seed > 0.5:
            # 替换head
            head = triple[0]
            rand_head = head
            while (rand_head == head):
                rand_head = random.randint(0, len(self.entity) - 1)  # 从实体集合随机选择一个实体作为头实体替换
            corrupted_triple[0] = rand_head

        else:
            # 替换tail
            tail = triple[1]
            rand_tail = tail
            while (rand_tail == tail):
                rand_tail = random.randint(0, len(self.entity) - 1)  # 从实体集合随机选择一个实体作为尾实体替换
            corrupted_triple[1] = rand_tail
        return corrupted_triple

    def update_embeddings(self, Tbatch):
        # 这里深拷贝了整个字典,会造成训练速度较慢，但是是易于理解的，增加训练速度的版本是transE_speed
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)
        for triple, corrupted_triple in Tbatch:
            # 取copy里的vector累积更新
            h_correct_update = copy_entity[triple[0]]  # 取正确三元组的头实体的（初始化）向量
            t_correct_update = copy_entity[triple[1]]  # 取正确三元组的尾实体的（初始化）向量
            relation_update = copy_relation[triple[2]]  # 取正确三元组的关系的（初始化）向量

            h_corrupt_update = copy_entity[corrupted_triple[0]]  # 取错误三元组的头实体的(初始化)向量
            t_corrupt_update = copy_entity[corrupted_triple[1]]  # 取错误三元组的尾实体的(初始化)向量

            # 取原始的vector计算梯度
            h_correct = self.entity[triple[0]]  # 正确三元组的头实体向量
            t_correct = self.entity[triple[1]]  # 正确三元组的尾实体向量
            relation = self.relation[triple[2]]  # 正确三元组的关系向量

            h_corrupt = self.entity[corrupted_triple[0]]  # 错误三元组的头实体向量
            t_corrupt = self.entity[corrupted_triple[1]]  # 错误三元组的尾实体向量

            if self.L1:  # 选择l1还是l2范数计算距离这是一个超参数
                dist_correct = distanceL1(h_correct, relation, t_correct)  # 计算正确三元组的L1范数
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)  # 计算错误三元组的L1范数
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)  # 计算正确三元组的L2范数
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)  # 计算错误三元组的L2范数

            err = self.hinge_loss(dist_correct, dist_corrupt)  # 计算三元组的距离函数，也是原文中的距离函数，目标是使正确三元组和错误三元组区分开

            if err > 0:  # 误差只可能是大于0和等于0,等于0时不能求导
                self.loss += err
                # 关于L1范数的求导方法：参考了刘知远组实现中的实现，是先对L2范数求导，逐元素判断正负，为正赋值为1，负则为-1。
                grad_pos = 2 * (h_correct + relation - t_correct)  # 正确三元组对L2范数求导的结果
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)  # 错误三元组对L2范数求导的结果

                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1
                '''
                update_embeddings函数中，要对correct triplet和corrupted triplet都进行更新。
                虽然写作$(h,l,t)$和$(h',l,t')$，但两个三元组只有一个entity不同（前面说了，不同时替换头尾实体），
                所以在每步更新时重叠实体需要更新两次（和更新relation一样）。
                例如正确的三元组是（1，2，3），错误的是（1，2，4），那么1和2都需要更新两次
                '''
                # 更新正确三元组的向量，更新的过程也就是梯度下降的过程
                # （h+r-t）head系数为正，减梯度；tail系数为负，加梯度
                # 基于正确三元组更新向量
                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos
                relation_update -= self.learning_rate * grad_pos  # 正确三元组的时候relation前面的符号是正号
                # corrupt项整体为负，因此符号与correct相反
                # 基于错误三元组更新向量
                if triple[0] == corrupted_triple[0]:  # 若替换的是尾实体，则头实体更新两次
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg
                    relation_update -= (-1) * self.learning_rate * grad_neg  # 错误三元组的时候relation前面的符号是负号
                elif triple[1] == corrupted_triple[1]:  # 若替换的是头实体，则尾实体更新两次
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg
                    relation_update -= (-1) * self.learning_rate * grad_neg  # 错误三元组的时候relation前面的符号是负号

        # batch norm，更新完一个epoch重新归一化一下
        for i in copy_entity.keys():  # copy_entity.keys代表实体的编号
            copy_entity[i] /= np.linalg.norm(copy_entity[i])  # l1范数归一化，这里好像跟原文不符，原文每个epoch前貌似用的l2范数归一化，而且只针对实体
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])  # l1范数归一化
        # 达到批量更新的目的
        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):  # 损失函数
        return max(0, dist_correct - dist_corrupt + self.margin)


if __name__ == '__main__':
    file1 = "../FB15k/"
    entity_set, relation_set, triple_list = data_loader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    # 论文中针对transe训练FB15k使用的超参数是k=50,embedding_dim=50, learning_rate=0.01, margin=1, L1范数
    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=400)
