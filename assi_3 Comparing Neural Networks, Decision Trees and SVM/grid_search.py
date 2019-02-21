from sklearn_svm import *
from FileOperation import *
import numpy as np
from multiprocessing import Process, Queue

class Worker(Process):
    def __init__(self, inQ, outQ):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ

    def run(self):
        while True:
            task = self.inQ.get()
            C, gamma, X, y, test, test_r = task
            clf = svm.SVC(C=C,
                          gamma=gamma,
                          cache_size=500)
            clf.fit(X, y)
            score = clf.score(test, test_r)
            self.outQ.put((C, gamma, score))

def createWorker(num):
    for i in range(num):
        workers.append(Worker(inQ, outQ))
        workers[i].start()

def destroyWorker():
    for w in workers:
        w.terminate()

def runGridSearch(fea = None, std = 0):
    flag = 'none '
    paraC = [5*i for i in range(1, 200)]
    paraGamma = [2*i for i in range(1,50)]
    X, y, test = getData('train.csv', 'test.csv')
    _, test_r = getData('test_t.csv')
    if isinstance(fea, int):
        X, test = np.delete(X, fea, 1), np.delete(test, fea, 1)
        flag += 'delc' + str(fea) + " "
    # 标准化
    # X, y  = LOFtransform(X, y)
    flag += "LOF "
    if std == 0:
        X, test = standardlization(StandardScaler, X, test)
        flag += "StandardScaler"
    elif std == 1:
        X, test = standardlization(MinMaxScaler, X, test)
        flag += "MinMaxScaler"
    elif std == 2:
        X, test = standardlization(RobustScaler, X, test)
        flag += "RobustScaler"
    elif std == 3:
        X, test = standardlization(QuantileTransformer, X, test)
        flag += "QuantileTransformer"
    elif std == 4:
        X0, test0 = standardlization(StandardScaler, X, test)
        X1, test1 = standardlization(MinMaxScaler, X, test)
        X, test = combine2(X0, X1, test0, test1, PCA=True)
        flag += "StandardScalerMinMaxScaler"
    elif std == 5:
        X0, test0 = standardlization(StandardScaler, X, test)
        X1, test1 = standardlization(MinMaxScaler, X, test)
        X2, test2 = standardlization(RobustScaler, X, test)
        X, test = combine3(X0, X1, X2, test0, test1, test2, PCA=True)
        flag += "StandardScalerMinMaxScalerRobustScaler"
    elif std == 6:
        X0, test0 = standardlization(StandardScaler, X, test)
        X1, test1 = standardlization(MinMaxScaler, X, test)
        X2, test2 = standardlization(RobustScaler, X, test)
        X3, test3 = standardlization(QuantileTransformer, X, test)
        X, test = combine4(X0, X1, X2, X3, test0, test1, test2, test3, PCA=True)
        flag += "StandardScalerMinMaxScalerRobustScalerQuantileTransformer"
    # X, test = combine4(X0, X1, X2, X3, test0, test1, test2, test3, PCA=True)
    result = []
    count = 0
    print("sklearn svm...")
    for C in paraC:
        for gamma in paraGamma:
            inQ.put((C, gamma, X, y, test, test_r))
            count += 1
    bestScore = 0
    bestC = 0
    bestGamma = 0
    for i in range(count):
        C, gamma, score = outQ.get()
        result.append([C, gamma, score])
        if score > bestScore:
            bestScore = score
            bestC = C
            bestGamma = gamma
            print(flag, "score = %f, C = %f, gamma = %f"%(bestScore, bestC, bestGamma))
    setData(result, flag)


if __name__ == '__main__':
    # for i in range(4):
    #     PROCESS = 48
    #     inQ = Queue()
    #     outQ = Queue()
    #     workers = []
    #     createWorker(PROCESS)
    #     runGridSearch(std=i)
    #     destroyWorker()
    # for i in (0, 1):
    #     PROCESS = 16
    #     inQ = Queue()
    #     outQ = Queue()
    #     workers = []
    #     createWorker(PROCESS)
    #     runGridSearch(fea=i)
    #     destroyWorker()
    PROCESS = 16
    inQ = Queue()
    outQ = Queue()
    workers = []
    createWorker(PROCESS)
    runGridSearch(std=0)
    destroyWorker()