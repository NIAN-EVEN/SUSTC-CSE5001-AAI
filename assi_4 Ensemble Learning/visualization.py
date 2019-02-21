import graphviz
from ensembleLearning import *
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

def scatterChart(X, y):
    attribute = ["SL", "Time", "BP", "Circulation", "HR", "EEG"]
    for attri in attribute:
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.set_title(attri)
        plt.xlabel('class')
        plt.ylabel(attri + '_value')
        maximum = X[attri].max()
        minimum = X[attri].min()
        plt.yticks(np.arange(minimum, maximum, (maximum - minimum) / 20))
        axis.scatter(y, X[attri], c='b', marker='.')
        plt.show()

def zhexiantu(X, y):
    # Xarray = X
    color = ['red', 'blue', 'maroon', 'green', 'cyan', 'deeppink']
    for i in range(len(X)):
        pass

def visualTree():
    trainfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\train.csv"
    testfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\test.csv"

    X, y, Xt = getData(trainfile, testfile)  # 获取数据
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)  # 训练模型
    dot_data = tree.export_graphviz(clf, out_file=None,
                     filled=True, rounded=True,
                     special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename="tree", directory="forest", format="pdf")

def result_ana():
    '''训练产生的数据分析及可视化'''
    # 获取所有数据
    rsl1 = pd.read_csv("result\\result1.csv")
    rsl2 = pd.read_csv("result\\result2.csv")
    # 获取对比的弱分类器数量数据
    ana_es1 = rsl1.loc[8019:8056:2, ["param_n_estimators", "mean_test_score"]]
    ana_es2 = rsl1.loc[39:70, ["param_n_estimators", "mean_test_score"]]
    ana_es = ana_es1.append(ana_es2)
    # 绘图
    plt.plot(ana_es["param_n_estimators"], ana_es["mean_test_score"])
    plt.xlabel("num of estimator")
    plt.ylabel("score")
    plt.show()

    # 数深度影响
    ana_max_depth = result.loc[5535:10782:228, ["param_max_depth", "mean_test_score"]]
    plt.plot(ana_max_depth["param_max_depth"], ana_max_depth["mean_test_score"])
    plt.xlabel("depth")
    plt.ylabel("score")
    plt.show()

    # 获取对比特征选择的数量
    ana_max_features = result.loc[8005:8197:38, ["param_max_features", "mean_test_score"]]
    # 绘图
    plt.plot(ana_max_features["param_max_features"], ana_max_features["mean_test_score"])
    plt.xlabel("feature number")
    plt.ylabel("score")
    plt.show()

    # 分类依据
    ana_gini = result.loc[0:5472, ["param_criterion", "mean_test_score"]]
    ana_entropy = result.loc[5472:10944, ["param_criterion", "mean_test_score"]]

    # 袋外数据评估
    ana_f = result.loc[0:10944:2, ["param_oob_score", "mean_test_score"]]
    ana_t = result.loc[1:10944:2, ["param_oob_score", "mean_test_score"]]

    # etResult
    etResult = pd.read_csv("result\\etResult1.csv")
    ana_spit = etResult[30000:30100]
    plt.plot(ana_spit["param_min_samples_split"], ana_spit["mean_test_score"])
    plt.xlabel("min_samples_split")
    plt.ylabel("score")
    plt.show()

    ana_leaf = etResult[30001:39902:100]
    plt.plot(ana_leaf["param_min_samples_leaf"], ana_leaf["mean_test_score"])
    plt.xlabel("min_samples_leaf")
    plt.ylabel("score")
    plt.show()

    ana_feature = etResult[1:30002:10000]
    plt.plot(ana_feature["param_max_features"], ana_feature["mean_test_score"])
    plt.xlabel("max_features")
    plt.ylabel("score")
    plt.show()

if __name__ == "__main__":
    trainfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\train.csv"
    testfile = "C:\\Users\\97439\\Desktop\\codes\\AAI\\AAI_assi_4\\test.csv"
    X, y, Xt = getData(trainfile, testfile)
    scatterChart(X, y)