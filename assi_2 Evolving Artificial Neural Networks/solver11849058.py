# -*- coding: utf-8 -*
'''
程序功能：解决5-Parity问题
程序输入：无
程序输出：一个可解决问题的神经网络
制作人：董广念11849058
'''

import numpy as np
import copy as cp
import math as mh
import time


start_time = time.time()
# M个隐藏层节点，来源: [1]TABLE 1
M = 20
# 初始化训练迭代次数, 再训练迭代次数
K0 = 2000
# 隐藏层节点最大最小值 2-5个，来源: [1]TABLE 1
NODES_RANGE = (9, 13)
# 初始连接密度
CD = 0.9
# 初始学习率
RATE_RANGE = (0.1, 1)
LR = 0.5
# 测试和训练数据集
TEST_DATA = []
INPUT_DATA = []
MAX_RUNNING_TIME = 17990
# 最大迭代次数，随便定的
MAX_GENERATION = 100

class ANN(object):
    def __init__(self, nodes_range = NODES_RANGE,cd = CD, lr = LR, input_nodes = 5):
        # 随机生成节点数量，节点权重
        self.learning_rate = lr
        self.input_num = input_nodes + 1 # 一个输入恒为-1的hidden_node与其他边都相连
        self.node_num = np.random.randint(*nodes_range)
        # self.node_num = 9
        self.hidden_out_num = self.node_num - self.input_num
        # self.bias = np.random.normal(size = (self.node_num-self.input_num))
        # 随机生成边数组，边权数组
        self.connection = np.zeros((self.node_num, self.hidden_out_num), dtype= bool)
        self.weight = np.zeros((self.node_num, self.hidden_out_num))
        for i in range(self.node_num):
            for j in range(self.hidden_out_num):
                if np.random.random() < cd and j > i-self.input_num:
                # if j > i - self.input_num:
                    self.connection[i][j] = True
                    # self.weight[i][j] = np.random.normal()
                    self.weight[i][j] = np.random.uniform(-1,1)

    # 计算得分函数
    def evaluate(self):
        '''神经网络的计算函数,由输入计算输出'''
        self.test_results = []
        self.feedforward()
        for num in self.output[:,-1]:
            self.test_results.append(1 if num>0.5 else 0)
        sum, count = 0, 0
        for x, y, z in zip(self.output, TEST_DATA, self.test_results):
            sum += (x[-1] - y[-1]) ** 2
            count += 1 if z == y[-1] else 0
        # self.score = 100 * sum / len(TEST_DATA)
        self.score, self.correct_num = sum, count

    # 后馈函数，从前向后计算输出值
    def feedforward(self):
        # output为网络中每一个节点的输出
        self.output = np.zeros([len(INPUT_DATA), self.node_num])
        # input_nodes层的输出为对应的输入
        self.output[:,0:self.input_num] = INPUT_DATA
        # 隐藏层和输出层的输出为相应的sigmod函数，i为第i个node
        for i in range(self.hidden_out_num):
            wx = np.dot(self.output,self.weight[:,i])
            self.output[:,self.input_num+i] = sigmod(wx)

    # modified BP函数，用于训练ANN
    def MBP(self, epoches = K0):
        '''反向传播函数，用于神经网络的参数调整'''
        # 训练epoches轮次
        for i in range(epoches):
            old_score = self.score
            self.weight -= self.learning_rate * self.backprop()
            self.evaluate()
            if self.correct_num == 2**(self.input_num-1) or self.score == old_score:
                break

    # 反向传播算法
    def backprop(self):
        '''反向传播计算函数，计算网络输出对所有节点边权和阈值的偏导数和'''
        # 误差函数对所有非输入节点输出的偏导
        deriv_o_net = self.output*(1-self.output)
        delta = np.zeros((len(TEST_DATA),self.hidden_out_num))
        t = [x[-1] for x in TEST_DATA]
        # delta[:,-1] = (200/len(TEST_DATA))*(self.output[:,-1] - t)*deriv_o_net[:,-1]
        delta[:, -1] = 2 * (self.output[:, -1] - t) * deriv_o_net[:, -1]
        for i in range(1,self.hidden_out_num):
            delta[:,-1-i] = np.dot(self.weight[-1-i],delta.T)*deriv_o_net[:,-1-i]
        deriv_E_w = np.dot(self.output.T, delta)*self.connection/len(TEST_DATA)
        # test2 = self.learning_rate * deriv_E_w
        return deriv_E_w

    def adam_backprop(self, epoches = K0):
        m = np.zeros_like(self.weight)
        v = np.zeros_like(self.weight)
        alpha = 0.1
        # beta_1 = 0.9
        beta_1 = 0.5
        beta_2 = 0.999
        epsilon = 1e-8
        t = 0
        for i in range(epoches):
            old_score = self.score
            gradient = self.backprop()
            t = t+1
            m = beta_1*m + (1-beta_1)*gradient
            v = beta_2*v + (1-beta_2)*gradient*gradient
            m_heat = m/(1-np.power(beta_1,t))
            v_heat = v/(1-np.power(beta_2,t))
            self.weight -= alpha * m_heat/(np.sqrt(v_heat)+epsilon)
            self.evaluate()
            if self.correct_num == 2**(self.input_num-1) or self.score == old_score:
                break
    def calculate_links(self):
        self.link = 0
        for i in range(self.node_num):
            for j in range(self.hidden_out_num):
                if self.connection[i,j]:
                    self.link+=1

    def output_structure(self):
        for j in range(self.hidden_out_num):
            for i in range(self.node_num - 1):
                print(self.weight[i,j],end=' ')
            print()

    def flyaway(self):
        for i in range(self.node_num):
            for j in range(self.hidden_out_num):
                if j>i-self.input_num:
                # if j > i - self.input_num:
                    self.connection[i][j] = True
                    # self.weight[i][j] = np.random.normal()
                    self.weight[i][j] = np.random.uniform(-1,1)

    def output_correct_rate(self):
        '''输出测试结果，并输出正确率'''
        count = 0
        idx = 0
        for z, x, y in zip(self.output[:,-1], self.test_results, TEST_DATA):
            idx+=1
            if x == y[-1]:
                count += 1
                print(idx, ":", z, '\t', x, y[-1])
            else:
                print(idx,":",z,'\t',x, y[-1], ' warning')
        print('CORRECT_NUM:', count, 'ERROR:', self.score)

#初始化pop为list类型
def setup(M):
    '''    初始化函数，用于ANN种群的初始化'''
    pop = []
    while(M>0):
        ann = ANN()
        ann.evaluate()
        ann.adam_backprop()
        pop.append(ann)
        M-=1
    pop.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
    return pop

def select(pop):
    '''基于排序的选择操作，用于选择种群中的一个个体进行变异操作'''
    range = len(pop)*(len(pop)+1)/2
    # rand为0-range的随机数
    rand = np.random.random()*range
    # 在整个range中j占比∑(1~j-1)-∑(1~j)部分
    # 对rand反向求是哪个数累加而成再加1即实现按照排序选择的功能
    j = int((mh.sqrt(8*rand+1)-1)/2)+1
    return pop[len(pop)-j]

def running(pop,birds):
    '''循环停止条件，用于判断是否应该停止循环'''
    # 到达限制时间
    if time.time() - start_time > MAX_RUNNING_TIME:
        return False
    # 得到想要的结果
    if pop[0].correct_num == 32 and pop[0].node_num <= 9:
        return False
    if birds[0].correct_num == 32 and birds[0].node_num <= 9:
        return False

    return True

def evolve(pop):
    ann = select(pop)  # 未测试
    if ann.correct_num != 2**(ann.input_num-1):
        offspring = simulate_anneal(ann)
        if offspring != ann:
            offspring.adam_backprop()
            pop[pop.index(ann)] = offspring
            return
    offspring = cut_node(ann)
    if offspring.score < pop[-1].score or offspring.correct_num > pop[-1].correct_num:
        pop[-1] = offspring
        return
    offspring = cut_edge(ann)
    if offspring.score < pop[-1].score or offspring.correct_num > pop[-1].correct_num:
        pop[-1] = offspring
        return
    # 增加边
    offspring = add_edge(ann)
    while offspring != None:
        if offspring.score < pop[-1].score or offspring.correct_num > pop[-1].correct_num:
            pop[-1] = offspring
            return
        else:
            offspring = add_edge(offspring)
    offspring = add_node(ann)
    if offspring.score < pop[-1].score or offspring.correct_num > pop[-1].correct_num:
        pop[-1] = offspring
        return

def simulate_anneal(ann):
    offspring = [ann]
    T = 100
    cool = 0.98
    while T>1e-8:# 1141次
        T = cool*T
        neighbour = get_neighbour(offspring[-1])
        delta = neighbour.score - offspring[-1].score
        if delta < 0 or np.random.random() < np.exp(-delta/T):
            offspring.append(neighbour)
    offspring.sort(key=lambda x: (32-x.correct_num, x.node_num, x.score))
    return offspring[0]

def get_neighbour(ann):
    neighbour = cp.deepcopy(ann)
    neighbour.weight += np.random.rand(*ann.weight.shape)*2-1
    neighbour.evaluate()
    return neighbour

def cut_node(ann):
    '''随机删除点'''
    idx = np.random.randint(ann.hidden_out_num-1)
    offspring = cp.deepcopy(ann)
    offspring.node_num -= 1
    offspring.hidden_out_num -= 1
    offspring.connection = np.delete(offspring.connection,idx,1)
    offspring.connection = np.delete(offspring.connection,offspring.input_num+idx,0)
    offspring.weight = np.delete(offspring.weight, idx, 1)
    offspring.weight = np.delete(offspring.weight, offspring.input_num + idx, 0)
    offspring.evaluate()
    offspring.adam_backprop()
    return offspring

def cut_edge(ann):
    '''随机删除边'''
    # 基本思想是，每条边权重绝对值越大待变这条边越重要，按照概率删除
    offspring = cp.deepcopy(ann)
    sum_weight = sum(sum(np.abs(offspring.weight)))
    selected_prob = np.abs(offspring.weight)/sum_weight
    prob = np.random.rand(*offspring.connection.shape)*offspring.connection-selected_prob
    x, y =np.where(prob == np.max(prob))
    offspring.connection[x,y] = False
    offspring.weight[x,y] = 0
    offspring.evaluate()
    offspring.adam_backprop()
    return offspring

# def add_edge(ann):
#     '''随机增加边'''
#     offspring = cp.deepcopy(ann)
#     prob = np.random.rand(*offspring.connection.shape) * (~offspring.connection)
#     max, x, y= 0, 0, 0
#     for i in range(len(prob)):
#         for j in range(len(prob[i])):
#             if prob[i,j] > max and j > i-offspring.input_num:
#                 max, x, y = prob[i][j], i, j
#     offspring.connection[x,y] = True
#     offspring.weight[x,y] = np.random.uniform(-1,1)
#     offspring.evaluate()
#     offspring.adam_backprop()
#     return offspring

def add_edge(ann):
    offspring = cp.deepcopy(ann)
    flag = False
    for i in range(offspring.node_num):
        for j in range(offspring.hidden_out_num):
            if j>i-offspring.input_num and offspring.connection[i,j] == False:
                offspring.connection[i,j] = True
                offspring.weight[i,j] = np.random.uniform(-1, 1)
                offspring.evaluate()
                offspring.adam_backprop()
                flag = True
                break
        if flag == True:
            break
    if flag == True:
        return offspring
    else:
        return None

def add_node(ann):
    '''随机删除点'''
    idx = np.random.randint(ann.hidden_out_num - 1)
    offspring = cp.deepcopy(ann)
    offspring.node_num += 1
    offspring.hidden_out_num += 1
    offspring.connection = np.insert(offspring.connection, idx, offspring.connection[:,idx],axis=1)
    offspring.connection = np.insert(offspring.connection, offspring.input_num + idx, offspring.connection[offspring.input_num + idx], axis=0)
    offspring.weight = np.insert(offspring.weight, idx, offspring.weight[:,idx],axis=1)
    offspring.weight = np.insert(offspring.weight, offspring.input_num + idx, offspring.weight[offspring.input_num + idx], axis=0)
    offspring.evaluate()
    offspring.adam_backprop()
    return offspring

def sigmod(x):
    return 1.0/(1.0+np.exp(-x))

def sigmod_deriv(x):
    return sigmod(x)*(1-sigmod(x))

def generate_DATA(x):
    '''产生x位的TEST_DATA'''
    TEST_DATA.clear()
    INPUT_DATA.clear()
    for i in range(2**x):
        input = list(map(int, bin(i)[2:].rjust(x, '0')))
        input.insert(0, -1)
        INPUT_DATA.append(input.copy())
        count = 0
        for c in input:
            if c == 1:
                count += 1
        # 作业要求版本
        if count%2==0:
            input.append(1)
        else:
            input.append(0)
        # # 测试版本
        # if count % 2 == 0:
        #     input.append(0)
        # else:
        #     input.append(1)
        TEST_DATA.append(input)

def setbirds():
    birds = []
    for node in range(*NODES_RANGE):
        bird = ANN(nodes_range=(node,node+1))
        bird.flyaway()
        bird.evaluate()
        bird.adam_backprop()
        birds.append(bird)
    birds.sort(key=lambda x: (32-x.correct_num, x.node_num, x.score))
    return birds

def flyaway(birds):
    for bird in birds:
        bird.flyaway()
        bird.evaluate()
        bird.adam_backprop()

def print_result(ann1,ann2):
    # 正规演化的经验上来说比随机游走的结构要好，所以线检查演化是否最优
    if ann1.correct_num==32 and ann1.node_num<=9:
        ann1.output_structure()
    # 烟花没有最优有两种情况，1，随机最优，2到时间了
    elif ann2.correct_num == 32 and ann2.node_num<=9:
        ann2.output_structure()
    # 到时间了，经验上说演化会更好
    else:
        ann1.output_structure()


if __name__ == "__main__":
    generate_DATA(5)
    # 初始化阶段
    pop = setup(M)
    birds = setbirds()
    # 演化阶段
    generation = 0
    while(running(pop, birds)):  # 未测试
        evolve(pop)
        flyaway(birds)
        birds.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
        pop.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
        generation += 1
    # 输出结果
    print_result(pop[0], birds[0])
    print(generation)
