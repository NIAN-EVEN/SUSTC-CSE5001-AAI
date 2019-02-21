from sklearn_svm import *
from FileOperation import *
from evolve_para import *

def simulate_anneal(para, X, y, test, test_r):
    T = 100
    cool = 0.98
    next = [para]
    while T>1e-8:
        T = cool * T
        neighbour = next[-1].mutation()
        neighbour.evaluate(X, y, test, test_r)
        delta = neighbour.score - next[-1].score
        if delta > 0 or np.random.random() < np.exp(-delta/T):
            next.append(neighbour)
        if delta > 0:
            print(neighbour)
    next.sort(key=lambda x: x.score, reverse=True)

def hill_climbing(para, X, y, test, test_r, step):
    ''' 基本思想，向上下左右四个方向走10%的本值步长，
     选择最好的方向前进，然后重复以上步骤
     如果没有方向可以使得得分变好，则退出'''
    direction = [0,0,0,0,0,0,0,0]
    while True:
        # 产生并计算四个方向的值
        newPara = [Parameter(para.get_C(), para.get_gamma(), flag='settled') for i in range(8)]
        newPara[0].set_C(para.get_C() * (1+step))
        newPara[1].set_C(para.get_C() * (1-step))
        newPara[2].set_gamma(para.get_gamma() * (1+step))
        newPara[3].set_gamma(para.get_gamma() * (1-step))
        newPara[4].set_C(para.get_C() * (1 + 0.5*step))
        newPara[4].set_gamma(para.get_gamma() * (1 + 0.5*step))
        newPara[5].set_C(para.get_C() * (1 - 0.5*step))
        newPara[5].set_gamma(para.get_gamma() * (1 - 0.5*step))
        newPara[6].set_C(para.get_C() * (1 - 0.5*step))
        newPara[6].set_gamma(para.get_gamma() * (1 + 0.5*step))
        newPara[7].set_C(para.get_C() * (1 + 0.5*step))
        newPara[7].set_gamma(para.get_gamma() * (1 - 0.5*step))
        for i in range(8):
            newPara[i].evaluate(X, y, test, test_r)
            print(newPara[i])
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

if __name__ == '__main__':
    X, y, test = getData('train_d.csv', 'test.csv')
    _, test_r = getData('test_t.csv')
    X, test = standardlization(X, test)
    X, test = np.delete(X, 0, 1), np.delete(test, 0, 1)
    para = Parameter(200, 220, 'settled')
    para.evaluate(X, y, test, test_r)
    print(para)
    step = [0.01, 0.02, 0.03, 0.05, 0.08]
    for s in step:
        hill_climbing(para, X, y, test, test_r, s)
        print(s, ' finish')