import numpy as np
import pandas as pd
import time, os, copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.externals import joblib


oriBound = {'SL': 522000.0, 'Time': 27053.0, 'BP': 207.0, 'Circulation': 18068.0, 'HR': 541.0, 'EEG': -15700.0}
def getData(trainfile, testfile):
    '''从文件中获取数据'''
    train = pd.read_csv(trainfile)
    Xt = pd.read_csv(testfile)
    col = [x for x in train.columns if x != 'Category']
    X = train[col]
    y = train['Category']
    return X, y, Xt
    # return np.array(X), np.array(y), np.array(Xt)

def setPrediction(filename, result):
    '''存储预测结果'''
    filename += ".csv"
    with open("result\\"+filename, 'x') as f:
        f.write("id,category\n")
        for idx, category in enumerate(result):
            f.write(str(idx+1) + ',' + str(category) + '\n')

def dataPreprocessing(X, Xt):
    '''数据预处理'''
    scl = QuantileTransformer().fit(X)
    return scl.transform(X), scl.transform(Xt)
    pass

def trainingModel(rfPara, etPara, adaPara, bgPara, X, y):
    '''构造模型并进行训练'''
    rf = RandomForestClassifier(n_estimators=rfPara["n_estimators"], criterion=rfPara["criterion"],
                                  max_depth=rfPara["max_depth"], max_features=rfPara["max_features"],
                                  oob_score=rfPara["oob_score"], n_jobs=rfPara["n_jobs"])

    et = ExtraTreesClassifier(n_estimators=etPara["n_estimators"], criterion=etPara["criterion"],
                              max_depth=etPara["max_depth"], max_features=etPara["max_features"],
                              min_samples_split=etPara["min_samples_split"], min_samples_leaf=etPara["min_samples_leaf"],
                              n_jobs=etPara["n_jobs"] )
    ada = AdaBoostClassifier(base_estimator=adaPara["base_estimator"], n_estimators=adaPara["n_estimators"],
                             learning_rate=adaPara["learning_rate"], algorithm=adaPara["algorithm"])

    bg = BaggingClassifier(base_estimator=bgPara["base_estimator"], n_estimators=bgPara["n_estimators"])

    expert = [("rf", rf), ("et", et), ("ada", ada), ("bg", bg)]

    return VotingClassifier(estimators=expert, voting="soft", n_jobs=-1).fit(X, y)

def singleModel(scaler, etPara, X, y):
    return scaler(n_estimators=etPara["n_estimators"], criterion=etPara["criterion"],
                  max_depth=etPara["max_depth"], max_features=etPara["max_features"],
                  min_samples_split=etPara["min_samples_split"], min_samples_leaf=etPara["min_samples_leaf"],
                  n_jobs=etPara["n_jobs"] ).fit(X, y)

if __name__ == "__main__":
    # 文件读取路径
    trainfile = "train.csv"
    testfile = "test.csv"
    # 参数
    rfPara = {
        "n_estimators": 180,    # 弱分类器的数量
        "criterion": "entropy", # 度量不纯度的标准，还可以是"entropy"
        "max_depth": None,        # 最大深度限制
        "max_features": 2,      # 每次分裂考虑的节点数量
        "oob_score": True,
        "n_jobs": -1            # 定义每个类的权重
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

    X, y, Xt = getData(trainfile, testfile)     # 获取数据
    # X, Xt = dataPreprocessing(X, Xt)
    clf = trainingModel(rfPara, etPara, adaParadt, bgPara, X, y)       # 训练模型
    # clf = singleModel(ExtraTreesClassifier, etPara, X, y)
    score = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print(score.mean())
    result = clf.predict(Xt)                    # 数据预测
    joblib.dump(clf, "trained_model")
    resultfile = "11849058-submission"
    setPrediction(resultfile, result)           # 预测结果存储
