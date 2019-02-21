import matplotlib.pyplot as plt
from ensembleLearning import *
from sklearn import tree


def verification4vote():
    '''随机森林运行100次查看实验结果'''
    start = time.time()
    # 文件读取路径
    trainfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\train.csv"
    testfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\test.csv"
    # 参数
    rfPara = {
        "n_estimators": 180,  # 弱分类器的数量
        "criterion": "entropy",  # 度量不纯度的标准，还可以是"entropy"
        "max_depth": None,  # 最大深度限制
        "max_features": 2,  # 每次分裂考虑的节点数量
        "oob_score": True,
        "n_jobs": -1  # 定义每个类的权重
    }

    etPara = {
        "n_estimators": 180,  # 弱分类器的数量
        "criterion": "gini",  # 度量不纯度的标准，还可以是"entropy"
        "max_depth": None,  # 最大深度限制
        "max_features": 5,  # 每次分裂考虑的节点数量
        "min_samples_split": 12,
        "min_samples_leaf": 1,
        "n_jobs": -1  # 定义每个类的权重
    }

    adaParadt = {
        "base_estimator": DecisionTreeClassifier(criterion="entropy", max_depth=19, max_features="log2"),
        "n_estimators": 180,  # 最大深度限制
        "learning_rate": 1.3,
        "algorithm": 'SAMME'
    }

    bgPara = {
        "base_estimator": DecisionTreeClassifier(criterion="entropy", max_depth=None, max_features="log2"),
        "n_estimators": 180,
    }

    X, y, Xt = getData(trainfile, testfile)  # 获取数据
    score = np.zeros(100)
    for i in range(100):
        clf = trainingModel(rfPara, etPara, adaParadt, bgPara, X, y)  # 训练模型
        # clf = singleModel(ExtraTreesClassifier, etPara, X, y)
        sc = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
        score[i] = sc.mean()
    print("mean:\t", score.mean())
    print("std:\t", score.std())
    print("var:\t", score.var())
    print("time:\t", time.time() - start)

def verificationEt():
    '''极限树运行100次查看实验结果'''
    start = time.time()
    # 文件读取路径
    trainfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\train.csv"
    testfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\test.csv"
    # 参数
    etPara = {
        "n_estimators": 180,  # 弱分类器的数量
        "criterion": "gini",  # 度量不纯度的标准，还可以是"entropy"
        "max_depth": None,  # 最大深度限制
        "max_features": 5,  # 每次分裂考虑的节点数量
        "min_samples_split": 12,
        "min_samples_leaf": 1,
        "n_jobs": -1  # 定义每个类的权重
    }
    X, y, Xt = getData(trainfile, testfile)  # 获取数据
    score = np.zeros(100)
    for i in range(100):
        clf = ExtraTreesClassifier(n_estimators=etPara["n_estimators"], criterion=etPara["criterion"],
                              max_depth=etPara["max_depth"], max_features=etPara["max_features"],
                              min_samples_split=etPara["min_samples_split"], min_samples_leaf=etPara["min_samples_leaf"],
                              n_jobs=etPara["n_jobs"] ).fit(X, y)
        # clf = singleModel(ExtraTreesClassifier, etPara, X, y)
        sc = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
        score[i] = sc.mean()
    print("mean:\t", score.mean())
    print("std:\t", score.std())
    print("var:\t", score.var())
    print("time:\t", time.time() - start)

def verificationRf():
    '''运行100次查看实验结果'''
    start = time.time()
    # 文件读取路径
    trainfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\train.csv"
    testfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\test.csv"
    # 参数
    rfPara = {
        "n_estimators": 180,  # 弱分类器的数量
        "criterion": "entropy",  # 度量不纯度的标准，还可以是"entropy"
        "max_depth": None,  # 最大深度限制
        "max_features": 2,  # 每次分裂考虑的节点数量
        "oob_score": True,
        "n_jobs": -1  # 定义每个类的权重
    }
    X, y, Xt = getData(trainfile, testfile)  # 获取数据
    score = np.zeros(100)
    for i in range(100):
        clf = RandomForestClassifier(n_estimators=rfPara["n_estimators"], criterion=rfPara["criterion"],
                                     max_depth=rfPara["max_depth"], max_features=rfPara["max_features"],
                                     oob_score=rfPara["oob_score"], n_jobs=rfPara["n_jobs"]).fit(X, y)
        # clf = singleModel(ExtraTreesClassifier, etPara, X, y)
        sc = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
        score[i] = sc.mean()
    print("mean:\t", score.mean())
    print("std:\t", score.std())
    print("var:\t", score.var())
    print("time:\t", time.time() - start)

def filter():
    '''筛选数据实验'''
    trainfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\train.csv"
    testfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\test.csv"
    gap = {"SL":0, "Time":0, "BP":0, "Circulation":0, "HR":0, "EEG":0}
    col = ["SL", "Time", "BP", "Circulation", "HR", "EEG"]
    X, y, Xt = getData(trainfile, testfile)  # 获取数据
    Xapos = np.array(X)
    for i, key in list(enumerate(gap))[:-1]:
        for j, data in enumerate(Xapos):
            # data的key属性比gap大，且data这条数据不是第0类
            if data[i] > gap[key] and y[j] != 0:
                gap[key] = data[i]
    for j, data in enumerate(Xapos):
        # data的key属性比gap小，且data这条数据不是第0类
        if data[5] < gap["EEG"] and y[j] != 0:
            gap["EEG"] = data[5]

    # 验证分位数正确性
    wrong_num = 0
    for i, data in enumerate(Xapos):
        for j in range(len(data)):
            if j != 5:
                if gap[col[j]] < data[j] and y[i] != 0:
                    wrong_num += 1
                    break
            else:
                if gap[col[j]] > data[j] and y[i] != 0:
                    wrong_num += 1
                    break
    # 筛选数据
    for cl in col[:-1]:
        Xapos = Xapos[Xapos[cl] < gap[cl]]
    Xapos = Xapos[Xapos["EEG"] > gap["EEG"]]

