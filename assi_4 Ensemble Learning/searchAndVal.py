from ensembleLearning import *
from sklearn.model_selection import GridSearchCV
from multiprocessing import Queue, Process

class Worker(Process):
    def __init__(self, inQ, outQ):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ

    def run(self):
        global X, y
        while True:
            task = self.inQ.get()
            n, l, X, y = task
            svc = SVC(C=215, gamma=4, cache_size=500)
            ab = AdaBoostClassifier(base_estimator=svc, n_estimators=n, learning_rate=l, algorithm="SAMME")
            score = cross_val_score(ab, X, y, cv=5, n_jobs=-1)
            self.outQ.put((n, l, score.mean()))

def createWorker(num):
    for i in range(num):
        workers.append(Worker(inQ, outQ))
        workers[i].start()

def destroyWorker():
    for w in workers:
        w.terminate()

def searchAdaBoost():
    global X, Xt
    createWorker(40)
    resultfile = "adaResultSvc.csv"
    parameters = {"n_estimators": range(10, 300, 10),  # 最大深度限制
                  "learning_rate": [0.05+0.05*x for x in range(40)]}
    X, Xt = dataPreprocessing(X, Xt)
    bestScore, best_es, best_lear, best_alg = 0, 0, 0, "SAMME"
    result = [["n_estimators", "learning_rate", "score"]]

    count = 0
    for n in range(10, 300, 10):
        for l in [0.05+0.05*x for x in range(40)]:
            inQ.put((n,l,X,y))
            count += 1

    for i in range(count):
        n, l, score = outQ.get()
        result.append([n, l, score])
        if score > bestScore:
            best_es = n
            best_lear = l
            print("es:", best_es, " lr:", best_lear, " score:", score.mean())
    destroyWorker()
    pd.DataFrame(result).to_csv(resultfile)


def searchRf():
    resultfile = "rfResult1.csv"
    parameters = {"n_estimators": range(1, 300),
                  "criterion": ("entropy", "gini"),
                  "max_features": range(1, 7),
                  "max_depth": range(1, 29)}  # 每次分裂考虑的节点数量
    rf = RandomForestClassifier(oob_score=True, n_jobs=-1)
    clf = GridSearchCV(rf, parameters, cv=5, n_jobs=-1, return_train_score=True)
    clf.fit(X, y)
    result = pd.DataFrame(clf.cv_results_)
    result.to_csv(resultfile)

def searchEt():
    resultfile = "etResult1.csv"
    parameters = {"max_features": (2, 3, 4, 5),
                  "min_samples_split": range(2, 1000, 10),
                  "min_samples_leaf": range(1, 1000, 10)}  # 每次分裂考虑的节点数量
    rf = ExtraTreesClassifier(criterion="gini", n_jobs=-1, n_estimators=180,
                                max_depth=None)
    clf = GridSearchCV(rf, parameters, cv=5, n_jobs=-1, return_train_score=True)
    clf.fit(X, y)
    result = pd.DataFrame(clf.cv_results_)
    result.to_csv(resultfile)

if __name__ == "__main__":
    inQ = Queue()
    outQ = Queue()
    workers = []
    # 文件读取路径
    trainfile = "train.csv"
    testfile = "test.csv"
    X, y, Xt = getData(trainfile, testfile)  # 获取数据
    searchEt()
