import numpy as np

def QlearningLambda(self, stateObs0, action_id, stateObs1, increScore, totalScore, info):
    '''在old_state->action->new_state中构建QTable学习'''
    stateIdx0, sstate0 = self.transformObs(stateObs0)
    stateIdx1, sstate1 = self.transformObs(stateObs1)
    self.checkStateExist(stateIdx0)
    self.checkStateExist(stateIdx1)
    reward = self.transReward(increScore, totalScore, info)
    if info == "NO_WINNER":
        target = reward + self.gamma * np.max(self.QTable[stateIdx1]) - self.QTable[stateIdx0][action_id]
    else:
        target = reward

    self.ETable[stateIdx0][action_id] += 1

    for key in self.QTable:
        self.QTable[key] += (self.learning_rate * target * self.ETable[key])

        self.ETable[key] *= (self.gamma * self.lbd)