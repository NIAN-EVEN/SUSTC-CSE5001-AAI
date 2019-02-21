import numpy as np
import time, os

def getData(trainFile, testFile = None):
    '''从文件中读取训练数据和测试数据
       训练数据存储到train中，并划分成(X, y)
       测试数据存储到test中'''
    X = []
    y = []
    test = []
    print("get data...")
    with open(trainFile, "r") as f:
        for line in f.readlines()[1:]:
            train = list(map(float, line.strip().split(",")))
            X.append(train[:-1])
            y.append(int(train[-1]))
    if testFile == None:
        return np.array(X), np.array(y)
    else:
        with open(testFile, "r") as f:
            for line in f.readlines()[1:]:
                test.append(list(map(float, line.strip().split(","))))
        return np.array(X), np.array(y), np.array(test)

def setResult(result, file):
    '''对测试集合的训练数据输出到文件中
        result表示预测结果，file代表filename后缀，存储当前参数'''
    print("set data...")
    result_dir = os.getcwd()+"\\results\\"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    filename = file + ".csv"
    with open(result_dir + filename, 'x') as f:
        f.write('id,category\n')
        for idx, category in enumerate(result):
            f.write(str(idx+1) + "," + str(category) +"\n")

def setData(result, filename):
    filename += '.txt'
    with open(filename, 'x') as f:
        f.write('C,gamma,score\n')
        for r in result:
            f.write('%s,%s,%s\n' %(str(r[0]), str(r[1]),str(r[2])))
