import matplotlib.pyplot as plt
from sklearn_svm import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer


def scatterChart(X, y, i):
    '''绘制散点图：
        6个属性6张图，每张图不同属性标记不同颜色
        横坐标是类，纵坐标是节点数据'''
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_title('attribute_'+str(i))
    plt.xlabel('class')
    plt.ylabel('attribute_'+str(i)+'_value')
    maximum = max(X[:,i])
    minimum = min(X[:,i])
    plt.yticks(np.arange(minimum, maximum, (maximum-minimum)/10))
    axis.scatter(y, X[:,i], c='b', marker='.')
    plt.show()
    # plt.savefig('plot\\point_%s.png' % str(i))

def zhexiantu(X, y):
    x = np.arange(0,X.shape[1],1)
    for i in range(X.shape[0]):
        if y[i] == 0:
            color = 'red'
        elif y[i] == 1:
            color = 'blue'
        elif y[i] == 2:
            color = 'maroon'
        elif y[i] == 3:
            color = 'green'
        elif y[i] == 4:
            color = 'cyan'
        elif y[i] == 5:
            color = 'deeppink'
        plt.plot(x, X[i], color = color)
    plt.show()

def apartZhexiantu(X, y, cata, color):
    x = np.arange(0, X.shape[1], 1)
    for i in range(X.shape[0]):
        if y[i] == cata:
            plt.plot(x, X[i], color=color)
    plt.show()

def scatterChartAll(X, y):
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.set_title('X:attribute y:value color:class')
    axis.set_xlabel('class')
    axis.set_ylabel('attribute')
    axis.set_zlabel('number')
    color = ['red', 'blue', 'maroon', 'green', 'cyan', 'deeppink']
    for i in range(len(y)):
        d1 = np.array([y[i], y[i], y[i], y[i], y[i], y[i]])
        d2 = np.array([0,1,2,3,4,5])
        d3 = X[i]
        axis.scatter(d1, d2, d3, c=color[y[i]], marker='.')
    plt.show()

if __name__ == '__main__':
    X, y = getData('train.csv')
    # X = LOFtransform(X)
    # X = standardlization(QuantileTransformer, X)
    # X = standardlization(QuantileTransformer, X)
    for i in range(6):
        scatterChart(X, y, i)
