import numpy as np
import time, os, copy
from scipy.stats import levy
from sklearn import svm
from sklearn_svm import *
from FileOperation import *
from func_timeout import func_set_timeout

# 交叉验证限定时间3分钟
LIMITED_TIME = 180
PROCESS = 5
STOP_TIME = 18000
STOP_SCORE = 0.69
POP_NUM = 40
OFFSPRING_NUM = 20

class Parameter(object):
    def __init__(self, C = parameter['C'], gamma = parameter['gamma'], flag = 'random'):
        '''初始化所有参数'''
        self.yitaC = np.random.rand()
        self.yitaGamma = np.random.rand() * gamma
        if flag == 'random':
            self.set_C(C + self.yitaC * np.random.random())
            self.set_gamma(gamma + self.yitaGamma * np.random.random())
        else:
            self.set_C(C)
            self.set_gamma(gamma)
        # score初始值设置为0则训练超时的个体不会通过训练提升分数，自然被淘汰
        self.score = 0

    def evaluate(self, X, y, test_d, test_r):
        '''使用交叉验证方式评分'''
        clf = svm.SVC(kernel='rbf', C=self.get_C(), gamma=self.get_gamma())
        clf.fit(X, y)
        self.score = clf.score(test_d, test_r)

    def crossover(self, para):
        '''交叉操作'''
        return Parameter(para.get_C(), para.get_gamma()), Parameter(para.get_C(), self.get_gamma())

    def mutation(self):
        '''变异操作'''
        n = 2 # n的数值大小未确定
        tao = 1/np.sqrt(2*np.sqrt(n))
        tao1 = 1/np.sqrt(2*n)
        para = Parameter(self.get_C(), self.get_gamma(), 'settled')
        para.yitaC = self.yitaC*np.exp(tao*np.random.normal() + tao1*np.random.normal())
        para.yitaGamma = self.yitaGamma * np.exp(tao * np.random.normal() + tao1 * np.random.normal())
        para.set_C(para.get_C() + para.yitaC * levy().rvs())
        para.set_gamma(para.get_gamma() + para.yitaGamma * levy().rvs())
        return para

    def set_C(self, C):
        self.__C = C if C > 0 else 1e-8

    def get_C(self):
        return self.__C

    def set_gamma(self, gamma):
        self.__gamma = gamma if gamma > 0 else 1e-8

    def get_gamma(self):
        return self.__gamma

    def __str__(self):
        '''输出所有的参数'''
        return 'C = ' + str(self.get_C()) + '; gamma = ' + str(self.get_gamma()) + '; score = ' + str(self.score)
        pass

def hill_climbing(para, X, y, test_d, test_r, step = 3e-4):
    ''' 基本思想，向上下左右四个方向走10%的本值步长，
     选择最好的方向前进，然后重复以上步骤
     如果没有方向可以使得得分变好，则退出'''
    direction = [1,1,1,1]
    while True:
        # 产生并计算四个方向的值
        newPara = [Parameter(para.get_C(), para.get_gamma(), flag='settled') for i in range(4)]
        newPara[0].set_C(para.get_C() * (1+step))
        newPara[1].set_C(para.get_C() * (1-step))
        newPara[2].set_gamma(para.get_gamma() * (1+step))
        newPara[3].set_gamma(para.get_gamma() * (1-step))
        for i in range(4):
            newPara[i].evaluate(X, y, test_d, test_r)
        # 计算并求解下一个方向
        nextDirec = 0
        for i in range(len(direction)):
            direction[i] = newPara[i].score - para.score
            if direction[i] > direction[nextDirec]:
                nextDirec = i
        if direction[nextDirec] > 0:
            para.set_C(newPara[nextDirec].get_C())
            para.set_gamma(newPara[nextDirec].get_gamma())
            para.score = newPara[nextDirec].score
        else:
            break

def setup(num, process, X, y, test_d, test_r):
    '''产生首代的函数'''
    pop = []
    for i in range(num):
        para = Parameter()
        para.evaluate(X, y, test_d, test_r)
        hill_climbing(para, X, y, test_d, test_r)
        pop.append(para)
    pop.sort(key=lambda x: x.score, reverse=True)
    return pop

def select(pop, num):
    '''根据排位随机选择下一代的函数'''
    range = len(pop)*(len(pop)+1)/2
    selected = []
    while len(selected)<num:
        rand = np.random.random()*range
        idx = int((np.sqrt(8*rand+1)-1)/2)+1
        if idx>=len(pop) or selected.__contains__(pop[idx]):
            continue
        else:
            selected.append(pop[idx])
    return selected

def evolve(pop, X, y, test_d, test_r):
    parents = select(pop, OFFSPRING_NUM)
    for param in parents:
        para = param.mutation()
        para.evaluate(X, y, test_d, test_r)
        hill_climbing(para, X, y, test_d, test_r)
        pop.append(para)
    pop.sort(key=lambda x: x.score, reverse=True)
    while len(pop) > POP_NUM or pop[-1].score == 0:
        pop.pop()

def pureSVC(X, y, test_d, test_r, flag = 'pure'):
    start = time.time()
    pop = setup(POP_NUM, PROCESS, X, y, test_d, test_r)
    # 如果运行超过5小时或子代得分超过0.7则停止变异
    print('start evolve')
    best = 0
    while (time.time() - start <= STOP_TIME and pop[0].score < STOP_SCORE):
        if pop[0].score > best:
            print('C = %s, gamma = %s, score = %s' %(str(pop[0].get_C()), str(pop[0].get_gamma()), str(pop[0].score)))
            best = pop[0].score
        evolve(pop, X, y, test_d, test_r)
        if len(pop) == 0:
            pop = setup(POP_NUM, PROCESS, X, y, test_d, test_r)
    setEvolveData(pop, flag)
    return pop[0]

if __name__ == '__main__':
    X, y, test = getData()
    # pcaX, _ = PCAtransform(X, X)
    # lofX, lofy = LOFtransform(X, y)
    # lofpcaX, _ = PCAtransform(lofX, lofX)
    # pcay = copy.deepcopy(y)
    # lofpcay = copy.deepcopy(lofy)
    test_d, test_r = getTestData()

    pureSVC(X, y, test_d, test_r, 'pure')
    # pureSVC(pcaX, y, test_d, test_r, 'pca')
    # pureSVC(lofX, lofy, test_d, test_r, 'lof')
    # pureSVC(lofpcaX, lofy, test_d, test_r, 'lofpca')
