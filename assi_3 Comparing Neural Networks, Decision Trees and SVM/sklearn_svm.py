from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor as LOF
from FileOperation import *
import numpy as np

# parameter = {'C': 215,
#               'gamma': 4,
#               'cache_size': 500}
parameter = {'C': 20.024,
              'gamma': 5.74e-7,
              'cache_size': 500}


def LOFtransform(X, y, con = 0.002):
    '''去除训练集中的离群点'''
    print('LOF transform...')
    LOFX = []
    LOFy = []
    lof = LOF(contamination=con)
    rsl = lof.fit_predict(X)
    for i in range(len(rsl)):
        if rsl[i] == 1:
            LOFX.append(X[i])
            LOFy.append(y[i])
    return np.array(LOFX), np.array(LOFy)


def LOFclassTransform(X, y, con = 0.002):
    '''去除训练集中的离群点'''
    print('LOF transform...')
    LOFX = []
    LOFy = []
    classes = ([],[],[],[],[],[])
    for i in range(len(y)):
        classes[y[i]].append(X[i])
    lof = LOF(contamination=con)
    for i in range(len(classes)):
        rsl = lof.fit_predict(classes[i])
        for j in range(len(rsl)):
            if rsl[j] == 1:
                LOFX.append(classes[i][j])
                LOFy.append(i)
    return np.array(LOFX), np.array(LOFy)

def standardlization(scaler, X, *other):
    scl = scaler().fit(X)
    others = []
    others.append(scl.transform(X))
    for o in other:
        others.append(scl.transform(o))
    if len(others) > 1:
        return others
    else:
        return  others[0]

def PCAtransform(X, test):
    '''PCA methods'''
    print('PAC transform...')
    # sklearn pca
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(X)
    return pca.transform(X), pca.transform(test)

def SVC_predict(X, y, parameter, test, test_r):
    result = []
    print("sklearn svm...")
    # clf = svm.SVC()
    print("training start from %d"%(time.time() - start))
    clf = svm.SVC(C=parameter['C'],
                  gamma=parameter['gamma'],
                  cache_size=parameter['cache_size'])
    clf.fit(X, y)
    print("training finish at %d" % (time.time() - start))
    result.extend(list(clf.predict(test)))
    print("predict finish at %d" % (time.time() - start))
    print('score:', clf.score(test, test_r))
    return result

def discardCol(X, test, i):
    '''删除第i列，忽略time，第二列，1'''
    return np.delete(X, i, 1), np.delete(test, i, 1)

def combine2(X0, X1, test0, test1, PCA = False):
    if PCA == False:
        return np.hstack((X0, X1)), np.hstack((test0, test1))
    else:
        return PCAtransform(np.hstack((X0, X1)), np.hstack((test0, test1)))

def combine3(X0, X1, X2, test0, test1, test2, PCA = False):
    if PCA == False:
        return np.hstack((X0, X1, X2)), np.hstack((test0, test1, test2))
    else:
        return PCAtransform(np.hstack((X0, X1, X2)), np.hstack((test0, test1, test2)))

def combine4(X0, X1, X2, X3, test0, test1, test2, test3, PCA = False):
    if PCA == False:
        return np.hstack((X0, X1, X2, X3)), np.hstack((test0, test1, test2, test3))
    else:
        return PCAtransform(np.hstack((X0, X1, X2, X3)), np.hstack((test0, test1, test2, test3)))

if __name__ == "__main__":
    start = time.time()
    X, y, test = getData('train.csv', 'test.csv')
    _, test_r = getData('test_t.csv')
    X, y = LOFtransform(X, y)
    # X0, test0 = standardlization(StandardScaler, X, test)
    # X1, test1 = standardlization(MinMaxScaler, X, test)
    # X2, test2 = standardlization(RobustScaler, X, test)
    # X3, test3 = standardlization(QuantileTransformer, X, test)
    # X, test = combine4(X0, X1, X2, X3, test0, test1, test2, test3, PCA=True)
    result = SVC_predict(X, y, parameter, test, test_r)
    # setResult(result, 'test')