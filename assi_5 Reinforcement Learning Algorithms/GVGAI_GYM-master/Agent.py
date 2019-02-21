from random import randint
import numpy as np
import pandas as pd
import json

ACTIONID = {
    "ACTION_NIL": 0,
    "ACTION_USE": 1,
    "ACTION_LEFT": 2,
    "ACTION_RIGHT": 3,
}
OBJ = {
    "AIRCRAFT": 15741,
    "STONE": 7960,
    "ALIEN": 11372,
    "SAM": 858,
    "BOMB": 5202,
    "BOMBP1": 2908,
    "BOMBP2": 2271
}

class Agent():

    def __init__(self):
        self.name = "randomAgent"
        # 读模型
        self.read_json("11849058agent.json")

    def act(self, t, stateObs, actions):
        '''当前状态包括当前的时间'''
        stateIdx = self.transformObs(t, stateObs)
        self.checkStateExist(stateIdx)
        action = np.array(self.QTable[stateIdx])
        action_id = action.argmax()
        return action_id

    def checkStateExist(self, idx):
        '''判断当前状态是否存在，如果不存在则加入该状态'''
        if idx not in self.QTable:
            self.QTable[idx] = np.zeros(len(self.actions))

    def transformObs(self, ticks, stateObs):
        '''利用ticks和stateObs构造RL环境状态'''
        # 返回9x10全景
        state = self.getFullState(stateObs)
        stateIdx = str(ticks)
        # 如果没有飞机则到达游戏结束状态
        if not (state == OBJ["AIRCRAFT"]).any():
            return "TERMINAL"
        # 加入飞机位置
        posAircraft = np.where(state == OBJ["AIRCRAFT"])
        for pos in posAircraft:
            for num in pos:
                stateIdx += str(num)
        # 加入炸弹位置
        if (state == OBJ["BOMB"]).any():
            posBombs = np.where(state == OBJ["BOMB"])
            for pos in posBombs:
                for num in pos:
                    stateIdx += str(num)
        # 炸弹移速为0.5，所以某些时刻会分成两部分
        if (state == OBJ["BOMBP2"]).any():
            posBombs2 = np.where(state == OBJ["BOMBP2"])
            for pos in posBombs2:
                for num in pos:
                    stateIdx += str(num)
        # 子弹位置
        if (state == OBJ["SAM"]).any():
            posBullet = np.where(state == OBJ["SAM"])
            for pos in posBullet:
                for num in pos:
                    stateIdx += str(num)

        # if (state == OBJ["SAM"]).any():
        #     stateIdx += '0'
        # else:
        #     stateIdx += '1'

        return stateIdx

    def getFullState(self, stateObs):
        '''将stateObs转化成9x9的方格'''
        env = []
        # 原始状态为(90,10,4)ndarray, 按照行列将每(10,10,4)个元素求和作为一个方格的值
        for i in range(int(stateObs.shape[0] / 10)):
            env.append([stateObs[i * 10:i * 10 + 10, j * 10:j * 10 + 10, :].sum() for j in range(int(stateObs.shape[1] / 10))])
        return np.array(env) - np.array(env).min()

    def read_json(self, filename):
        with open(filename, "r") as f:
            RLDict = json.load(f)
        QTable = {}
        for key, value in RLDict["QTable"].items():
            QTable[key] = np.array(value)
        self.actions = RLDict["actions"]
        self.learning_rate = RLDict["learning_rate"]
        self.gamma = RLDict["gamma"]
        self.epsilon = RLDict["epsilon"]
        self.lbd = RLDict["lbd"]
        self.QTable = QTable


    def randomAct(self, stateObs, actions):
        action_id = randint(0, len(actions) - 1)
        return action_id
