import numpy as np
import codecs
import operator
import json
from transE_simple import data_loader, entity2id, relation2id
# from transE_speed import data_loader, entity2id, relation2id
import time


def dataloader(entity_file, relation_file, test_file):
    # entity_file: entity \t embedding
    entity_dict = {}
    relation_dict = {}
    test_triple = []

    with codecs.open(entity_file) as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity, embedding = line.strip().split('\t')  # 获得实体和代表的向量
            embedding = np.array(json.loads(embedding))
            entity_dict[int(entity)] = embedding  # 将实体和向量一一对应成字典格式

    with codecs.open(relation_file) as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation, embedding = line.strip().split('\t')  # 获得关系和代表的向量
            embedding = np.array(json.loads(embedding))
            relation_dict[int(relation)] = embedding  # 将关系和向量一一对应成字典格式

    with codecs.open(test_file) as t_f:
        lines = t_f.readlines()
        for line in lines:
            triple = line.strip().split('\t')  # 获得测试集数据
            if len(triple) != 3:  # 找到头实体-关系-尾实体的三元组作为测试集
                continue
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            test_triple.append(tuple((h_, t_, r_)))

    return entity_dict, relation_dict, test_triple  # 获得实体、关系，测试集


def distance(h, r, t):
    return np.linalg.norm(h + r - t)


class Test:
    def __init__(self, entity_dict, relation_dict, test_triple, train_triple, isFit=True):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        print(len(self.entity_dict), len(self.relation_dict), len(self.test_triple), len(self.train_triple))
        self.isFit = isFit

        self.hits10 = 0
        self.mean_rank = 0

        self.relation_hits10 = 0
        self.relation_mean_rank = 0

    def rank(self):
        hits = 0
        rank_sum = 0
        step = 1
        start = time.time()
        for triple in self.test_triple:
            rank_head_dict = {}
            rank_tail_dict = {}

            for entity in self.entity_dict.keys():  # triple是测试集的三元组，entity是导入的实体,keys()是0123456。。。等数字标签
                if self.isFit:  # 测试时在替换后要检查一下新三元组是否出现在训练集中，是的话就删掉，这就是filter训练方法
                    if [entity, triple[1], triple[2]] not in self.train_triple:  # 测试集的尾实体、关系推理头实体
                        h_emb = self.entity_dict[entity]
                        r_emb = self.relation_dict[triple[2]]
                        t_emb = self.entity_dict[triple[1]]
                        rank_head_dict[entity] = distance(h_emb, r_emb, t_emb)  # distance  头+关系-尾
                else:  # 测试时在替换后不检查一下新三元组是否出现在训练集中，这是raw的训练方法
                    h_emb = self.entity_dict[entity]
                    r_emb = self.relation_dict[triple[2]]
                    t_emb = self.entity_dict[triple[1]]
                    rank_head_dict[entity] = distance(h_emb, r_emb, t_emb)

                if self.isFit:
                    if [triple[0], entity, triple[2]] not in self.train_triple:  # 测试集的头实体、关系推理尾实体
                        h_emb = self.entity_dict[triple[0]]
                        r_emb = self.relation_dict[triple[2]]
                        t_emb = self.entity_dict[entity]
                        rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[triple[0]]
                    r_emb = self.relation_dict[triple[2]]
                    t_emb = self.entity_dict[entity]
                    rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)
            # sorted(iterable, key=None, reverse=False) ，key -- 主要是用来进行比较的元素
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))

            # rank_sum and hits
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            step += 1
            if step % 200 == 0:
                end = time.time()
                print("step: ", step, " ,hit_top10_rate: ", hits / (2 * step), " ,mean_rank ", rank_sum / (2 * step),
                      'time of testing one triple: %s' % (round((end - start), 3)))
                start = end
        self.hits10 = hits / (2 * len(self.test_triple))
        self.mean_rank = rank_sum / (2 * len(self.test_triple))

    def relation_rank(self): # 多数论文中并没有对于关系排名的介绍，是这个代码自己写的，实体的在下面，换成实体可能分数会降低
        hits = 0
        rank_sum = 0
        step = 1

        start = time.time()
        for triple in self.test_triple:
            rank_dict = {}
            for r in self.relation_dict.keys():
                if self.isFit and (triple[0], triple[1], r) in self.train_triple:
                    continue
                h_emb = self.entity_dict[triple[0]]
                r_emb = self.relation_dict[r]
                t_emb = self.entity_dict[triple[1]]
                rank_dict[r] = distance(h_emb, r_emb, t_emb)

            rank_sorted = sorted(rank_dict.items(), key=operator.itemgetter(1))  # operator模块提供的itemgetter函数主要用于获取某一对象特定维度的数据
            # 这里根据rank_dict字典的分数来进行排名
            rank = 1
            for i in rank_sorted:
                if triple[2] == i[0]:# 预测对了，就跳出循环
                    break
                rank += 1
            if rank < 10:
                hits += 1
            rank_sum = rank_sum + rank + 1  # 没太懂为什么加1

            step += 1  # 每一个测试三元组step加1
            if step % 200 == 0:
                end = time.time()
                print("step: ", step, " ,hit_top10_rate: ", hits / step, " ,mean_rank ", rank_sum / step,
                      'used time: %s' % (round((end - start), 3)))
                start = end

        self.relation_hits10 = hits / len(self.test_triple)
        self.relation_mean_rank = rank_sum / len(self.test_triple)


if __name__ == '__main__':
    _, _, train_triple = data_loader("../FB15k/")  # 读取train_triple是为了filter训练方法

    entity_dict, relation_dict, test_triple = \
        dataloader("../res/entity_50dim_batch400", "../res/relation_50dim_batch400",
                   "../FB15k/test.txt")
    # dataloader("..\\res\\entity_temp_260epoch","..\\res\\relation_temp_260epoch",
    #            "..\\FB15k\\test.txt")

    test = Test(entity_dict, relation_dict, test_triple, train_triple, isFit=False)

    test.relation_rank()
    print("relation hits@10: ", test.relation_hits10)
    print("relation meanrank: ", test.relation_mean_rank)

    print("替换三元组的头和尾需要更多的时间...")
    test.rank()
    print("entity hits@10: ", test.hits10)
    print("entity meanrank: ", test.mean_rank)

    f = open("../res/result.txt", 'w')
    f.write("entity hits@10: " + str(test.hits10) + '\n')
    f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    f.close()
