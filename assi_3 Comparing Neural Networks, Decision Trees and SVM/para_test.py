from FileOperation import *
from sklearn_svm import *

def test_featureDelete(parameter, X0, y, test0, test_r, fea):
    X, test = np.delete(X0, fea, 1), np.delete(test0, fea, 1)
    scoring(parameter, X, y, test, test_r)


def scoring(parameter, X, y, test, test_r):
    print("sklearn svm...")
    start = time.time()
    clf = svm.SVC(C=parameter['C'],
                  gamma=parameter['gamma'],
                  cache_size=parameter['cache_size'],
                  class_weight=None)

    clf.fit(X, y)
    print('C: ', parameter['C'], 'gamma: ', parameter['gamma'], 'score:', clf.score(test, test_r),
          ' time: %d' % (time.time() - start))

def PCA(parameter, X0, y, test0, test_r):
    X, test = PCAtransform(X0, test0)
    scoring(parameter, X, y, test, test_r)


if __name__ == '__main__':
    parameter = {'C': 214.5,
                 'gamma': 4,
                 'cache_size': 500}
    c_weight = {0:0.283, 1:0.031, 2:0.152, 3:0.220, 4:0.215, 5:0.099}
    X, y, test= getData('train_d.csv', 'test.csv')
    _, test_r = getData('test_t.csv')
    X0, test0 = standardlization(X, test)
    X1, test1 = MaxMinStandardlization(X, test)
    X2, test2 = robustStandardlization(X, test)
    X3, test3 = qtStandardlization(X, test)
    X, test = combine4(X0, X1, X2, X3, test0, test1, test2, test3, PCA=True)
    # 运行删除特征值
    # test_featureDelete(parameter, X, y, test, test_r, 0)
    # PCA(parameter, X, y, test, test_r)
    scoring(parameter, X, y, test, test_r)
