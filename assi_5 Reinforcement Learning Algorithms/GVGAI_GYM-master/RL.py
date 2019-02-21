#!/usr/bin/env python
import gym
import numpy as np
import pandas as pd
from random import randint
import json
import gym_gvgai

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
    "BOMBP1": 2918,
    "BOMBP2": 2271
}

class RL(object):
    def __init__(self, actions, learning_rate, gamma, epsilon, lbd):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.lbd = lbd
        self.QTable = {}
        self.ETable = {}
        self.score = 0
        self.next_action = 1

    def checkStateExist(self, idx):
        '''判断当前状态是否存在，如果不存在则加入该状态'''
        if idx not in self.QTable:
            self.QTable[idx] = np.zeros(len(self.actions))
        if idx not in self.ETable:
            self.ETable[idx] = np.zeros(len(self.actions))

    def chooseAction(self, stateObs, probChoose=False, random=False):
        '''选择action操作
            random==True时为随机动作
            probChoose==True时为以得分为概率选择action'''
        if random:
            action_id = self.randomAction()
        else:
            action_id = self.epsilonGreedy(stateObs, probChoose)
        return action_id

    def randomAction(self):
        return randint(0, len(self.actions) - 1)

    def epsilonGreedy(self, stateObs, probChoose=False):
        '''epsilon-greedy算法, 弱probChoose==True则按value大小为概率选择action'''
        stateIdx, state = self.transformObs(stateObs)
        self.checkStateExist(stateIdx)
        if np.random.uniform() < self.epsilon:
            action_id = self.randomAction()
        else:
            action = self.QTable[stateIdx]
            # epsilon-greedy算法，概率选择行动
            if probChoose:
                # 以action value大小/总和为概率选择action
                action_id = probabilyty_pick(action)
            else:
                # 随机选择最大概率的action
                action_id = action.argmax()
                # action_id = np.random.choice(np.where(action == action.max())[0])
        return action_id

    def Qlearning(self, stateObs0, action_id, stateObs1, increScore, totalScore, info):
        '''在old_state->action->new_state中构建QTable学习'''
        stateIdx0, sstate0 = self.transformObs(stateObs0)
        stateIdx1, sstate1 = self.transformObs(stateObs1)
        self.checkStateExist(stateIdx0)
        self.checkStateExist(stateIdx1)
        reward = self.transReward(increScore, totalScore, info)
        if info == "NO_WINNER":
            target = reward + self.gamma * self.QTable[stateIdx1].max() - self.QTable[stateIdx0][action_id]
        else:
            target = reward
        self.QTable[stateIdx0][action_id] += (self.learning_rate * target)

    def SarsaLambda(self, stateObs0, action_id, stateObs1, increScore, totalScore, info):
        '''在old_state->action->new_state中构建QTable学习'''
        stateIdx0, sstate0 = self.transformObs(stateObs0)
        stateIdx1, sstate1 = self.transformObs(stateObs1)
        self.checkStateExist(stateIdx0)
        self.checkStateExist(stateIdx1)

        self.next_action = self.chooseAction(stateObs1)

        reward = self.transReward(increScore, totalScore, info)
        if info == "NO_WINNER":
            target = reward + self.gamma * self.QTable[stateIdx1][self.next_action] - self.QTable[stateIdx0][action_id]
        else:
            target = reward

        self.ETable[stateIdx0][action_id] += 1

        for key in self.QTable:
            self.QTable[key] += (self.learning_rate * target * self.ETable[key])

            self.ETable[key] *= (self.gamma * self.lbd)

    def transformObs(self, stateObs):
        '''从stateObs获取需要的状态, 并返回对应哈希值作为index'''
        state = self.getFullState(stateObs)
        # 如果不存在飞机，说明被摧毁，进入终止状态
        if not (state == OBJ["AIRCRAFT"]).any():
            return "TERMINAL", None
        pos = np.where(state == OBJ["AIRCRAFT"])
        rgMin = [int(pos[0])-8, int(pos[1])-1]
        rgMax = [int(pos[0]), int(pos[1])+1]
        if rgMin[1] < 0:
            rgMin[1] = 0
        if rgMax[1] > state.shape[1]-1:
            rgMax[1] = state.shape[1]
        shrinkedState = state[rgMin[0]:rgMax[0]+1, rgMin[1]:rgMax[1]+1]
        return str(shrinkedState), shrinkedState
    
    def getFullState(self, stateObs):
        '''将stateObs转化成9x9的方格'''
        env = []
        # 原始状态为(90,10,4)ndarray, 按照行列将每(10,10,4)个元素求和作为一个方格的值
        for i in range(int(stateObs.shape[0] / 10)):
            env.append([stateObs[i * 10:i * 10 + 10, j * 10:j * 10 + 10, :].sum() for j in range(int(stateObs.shape[1] / 10))])
        return np.array(env) - np.array(env).min()

    def transReward(self, increScore, totalScore, info):
        '''reward转换函数'''
        if info == "PLAYER_WINS":
            baseReward = totalScore
            if totalScore < self.score:
                reward = baseReward
            else:
                reward = baseReward + np.exp(np.abs(totalScore - self.score))
        elif info == "PLAYER_LOSES":
            if totalScore > self.score:
                reward = increScore
            else:
                reward = increScore - np.exp(np.abs(self.score - totalScore))
        else:
            reward = increScore**3
        return reward

    def reset(self):
        for key in self.ETable:
            self.ETable[key] *= 0

    def to_json(self, filename):
        QTable = {}
        for key, value in self.QTable.items():
            QTable[key] = value.tolist()
        RLDict = {
            "actions": self.actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "lbd": self.lbd,
            "QTable": QTable,
        }
        with open(filename, "w") as f:
            json.dump(RLDict, f)

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

def probabilyty_pick(elems):
    '''按elems中元素正值的大小为概率选择elem，返回被选元素id'''
    rand = np.random.uniform(0, max(elems))
    cumulitive_prob = 0
    for idx, prob in enumerate(elems):
        # rand落在min到max的区间来完成概率选择
        if prob > 0 and cumulitive_prob < rand and cumulitive_prob+prob > rand:
            return idx
        cumulitive_prob += prob
    # 如果没有正值则返回随机数
    return randint(0, len(elems)-1)


def runGame(env, agent):
    '''R为游戏迭代次数，filename为读取文件路径，withHistory为判断是否需要读取文件
       old_round用于观察'''
    stateObs = env.reset()
    env.render()
    ticks = 0
    totalScore = 0
    for t in range(2000):
        ticks += 1
        action_id = agent.chooseAction(stateObs)
        stateObsNew, increScore, done, debug = env.step(action_id)
        env.render()
        totalScore += increScore
        agent.Qlearning(stateObs, action_id, stateObsNew, increScore, totalScore, debug["winner"])
        stateObs = stateObsNew
        if done:
            if debug["winner"] =="PLAYER_WINS":
                print("WIN!!!")
            break
    return totalScore, ticks


if __name__ == "__main__":
    AGENT = "11849058Qlearning.json"
    ROUND = 3000
    RESULT = "11849058Qlearning.csv"
    col = ["reward", "ticks"]

    LEARNING_RATE = 0.9
    GAMMA = 0.9
    LAMBDA = 0.9
    MAX_EPSILON = 1
    MIN_EPSILON = 0.01
    DE_EPSILON = 0.01
    EPSILON = 1

    # 新建游戏，新建agent
    env = gym_gvgai.make('gvgai-aai-lvl0-v0')
    print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
    actions = env.env.GVGAI.actions()
    agent = RL(actions, LEARNING_RATE, GAMMA, EPSILON, LAMBDA)

    # 是否读取历史文件
    # agent.read_json(AGENT)

    totalReward = 0
    statistics50reward = 0
    result = []
    # 迭代训练R轮
    for round in range(ROUND):
        agent.reset()
        reward, ticks = runGame(env, agent)
        totalReward += reward
        statistics50reward += reward
        result.append([reward, ticks])
        agent.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DE_EPSILON*round)
        if round%20==0:
            print("round:%d" % round)
            agent.score = statistics50reward/20
            statistics50reward = 0

    # 存下agent信息
    agent.to_json(AGENT)

    # 存储结果信息
    pd.DataFrame(result, columns=col).to_csv(RESULT)
