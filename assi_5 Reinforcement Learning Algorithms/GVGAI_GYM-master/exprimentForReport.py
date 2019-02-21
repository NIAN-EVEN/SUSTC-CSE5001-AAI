from RL import *
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
    "BOMBP1": 2908,
    "BOMBP2": 2271
}
LEARNING_RATE = 0.99
GAMMA = 0.99
EPSILON = 0.02

class fullObsRl(RL):
    def __init__(self, actions, learning_rate, gamma, e_greedy):
        super(fullObsRl, self).__init__(actions, learning_rate, gamma, e_greedy)

    def transformObs(self, stateObs):
        state = self.getFullState(stateObs)
        return str(state), state

def runGame(env, agent, ROUND=1200, recmd=0.0, type="REWARD"):
    result = []
    for round in range(ROUND):
        reward = 0
        ticks = 0
        stateObs = env.reset()
        for t in range(2000):
            action_id = agent.chooseAction(stateObs, random=False, recommendProb=recmd)
            stateObsNew, increScore, done, debug = env.step(action_id)
            agent.learn(stateObs, action_id, stateObsNew, increScore, debug["winner"], type=type)
            stateObs = stateObsNew
            reward += increScore
            if done:
                break
            ticks += 1
        result.append([round, reward, ticks])
        if round%100==0:
            print("round: ", round)
    return result

def fullObsVsPartObs():
    # 参数设置
    col = ["round", "reward", "ticks"]
    fullResultFile = "model/fullObs.csv"
    partResultFile = "model/partObs.csv"
    fullModelFile = "model/full.json"
    partModelFile = "model/part.json"


    # 环境和learner初始化
    env = gym_gvgai.make('gvgai-aai-lvl0-v0')
    print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
    actions = env.env.GVGAI.actions()
    partAgent = RL(actions, LEARNING_RATE, GAMMA, EPSILON)
    fullAgent = fullObsRl(actions, LEARNING_RATE, GAMMA, EPSILON)

    print("full vs part is going on...")
    # 运行试验
    fullResult = runGame(env, fullAgent)
    partResult = runGame(env, partAgent)

    # 存储模型
    partAgent.to_json(partModelFile)
    fullAgent.to_json(fullModelFile)

    # 存储实验数据
    pd.DataFrame(fullResult, columns=col).to_csv(fullResultFile)
    pd.DataFrame(partResult, columns=col).to_csv(partResultFile)

def withRecommedVsWithout():
    # 参数设置
    col = ["round", "reward", "ticks"]
    filename = "model/rcmd_real"

    # 环境和learner初始化
    env = gym_gvgai.make('gvgai-aai-lvl0-v0')
    print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
    actions = env.env.GVGAI.actions()

    print("with or without rcmd is going on...")
    agent03 = RL(actions, LEARNING_RATE, GAMMA, EPSILON)
    agent06 = RL(actions, LEARNING_RATE, GAMMA, EPSILON)
    agent09 = RL(actions, LEARNING_RATE, GAMMA, EPSILON)

    # 带推荐训练
    runGame(env, agent03, recmd=0.3)
    runGame(env, agent06, recmd=0.6)
    runGame(env, agent09, recmd=0.9)

    agent03.to_json(filename + "03" + ".json")
    agent06.to_json(filename + "06" + ".json")
    agent09.to_json(filename + "09" + ".json")

    # 不带推荐验证
    result03 = runGame(env, agent03, recmd=0, ROUND=200)
    result06 = runGame(env, agent06, recmd=0, ROUND=200)
    result09 = runGame(env, agent09, recmd=0, ROUND=200)

    pd.DataFrame(result03, columns=col).to_csv(filename + "03" + ".csv")
    pd.DataFrame(result06, columns=col).to_csv(filename + "06" + ".csv")
    pd.DataFrame(result09, columns=col).to_csv(filename + "09" + ".csv")

def paneltyVsReward():
    # 参数设置
    col = ["round", "reward", "ticks"]
    filename = "model/PENALTY"

    # 环境和learner初始化
    env = gym_gvgai.make('gvgai-aai-lvl0-v0')
    print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
    actions = env.env.GVGAI.actions()
    agent = RL(actions, LEARNING_RATE, GAMMA, EPSILON)

    print("penalty is going on...")
    result = runGame(env, agent, type="PENALTY")
    agent.to_json(filename + ".json")
    pd.DataFrame(result, columns=col).to_csv(filename + ".csv")


if __name__ =="__main__":
    withRecommedVsWithout()
    # paneltyVsReward()
    # fullObsVsPartObs()

