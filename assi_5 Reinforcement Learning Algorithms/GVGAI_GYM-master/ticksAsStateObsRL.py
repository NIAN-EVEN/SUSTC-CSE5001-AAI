from RL import *

class tRL(RL):
    def __init__(self, actions, learning_rate, gamma, epsilon, lbd):
        super(tRL, self).__init__(actions, learning_rate, gamma, epsilon, lbd)

    def transformObs(self, ticks, stateObs):
        '''利用ticks和stateObs构造RL环境状态'''
        state = self.getFullState(stateObs)
        stateIdx = str(ticks)
        if not (state == OBJ["AIRCRAFT"]).any():
            return "TERMINAL"
        # 拿到飞机位置
        posAircraft = np.where(state == OBJ["AIRCRAFT"])
        for pos in posAircraft:
            for num in pos:
                stateIdx += str(num)
        # 拿到炸弹位置
        if (state == OBJ["BOMB"]).any():
            posBombs = np.where(state == OBJ["BOMB"])
            for pos in posBombs:
                for num in pos:
                    stateIdx += str(num)
        # 拿到炸弹位置
        if (state == OBJ["BOMBP2"]).any():
            posBombs2 = np.where(state == OBJ["BOMBP2"])
            for pos in posBombs2:
                for num in pos:
                    stateIdx += str(num)
        # 拿到子弹位置
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

    def chooseAction(self, ticks,  stateObs, probChoose=False, random=False):
        '''选择action操作
            random==True时为随机动作
            probChoose==True时为以得分为概率选择action'''
        if random:
            action_id = self.randomAction()
        else:
            action_id = self.epsilonGreedy(ticks, stateObs, probChoose)
        return action_id

    def epsilonGreedy(self, ticks, stateObs, probChoose=False):
        '''epsilon-greedy算法, 弱probChoose==True则按value大小为概率选择action'''
        stateIdx = self.transformObs(ticks, stateObs)
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

    def SarsaLambda(self, ticks, stateObs0, action_id, stateObs1, increScore, totalScore, info):
        '''在old_state->action->new_state中构建QTable学习'''
        stateIdx0 = self.transformObs(ticks, stateObs0)
        stateIdx1 = self.transformObs(ticks+1, stateObs1)
        self.checkStateExist(stateIdx0)
        self.checkStateExist(stateIdx1)

        self.next_action = self.chooseAction(ticks+1, stateObs1)

        reward = self.transReward(increScore, totalScore, info)
        if info == "NO_WINNER":
            target = reward + self.gamma * self.QTable[stateIdx1][self.next_action] - self.QTable[stateIdx0][action_id]
        else:
            target = reward

        self.ETable[stateIdx0] *= 0
        self.ETable[stateIdx0][action_id] = 1

        for key in self.QTable:
            self.checkStateExist(key)
            self.QTable[key] += (self.learning_rate * target * self.ETable[key])

            self.ETable[key] *= (self.gamma * self.lbd)

def runGame(env, agent):
    '''R为游戏迭代次数，filename为读取文件路径，withHistory为判断是否需要读取文件
       old_round用于观察'''
    stateObs = env.reset()
    env.render()
    ticks = 0
    totalScore = 0
    for t in range(2000):
        action_id = agent.next_action
        stateObsNew, increScore, done, debug = env.step(action_id)
        env.render()
        totalScore += increScore
        agent.SarsaLambda(ticks, stateObs, action_id, stateObsNew, increScore, totalScore, debug["winner"])
        stateObs = stateObsNew
        ticks += 1
        if done:
            if debug["winner"] =="PLAYER_WINS":
                print("WIN!!!")
            break
    return totalScore, ticks

if __name__ == "__main__":
    AGENT = "model/11849058submit.json"
    ROUND = 60
    RESULT = "model/11849058submit.csv"
    col = ["reward", "ticks"]

    LEARNING_RATE = 0.9
    GAMMA = 0.9
    LAMBDA = 0.9
    MAX_EPSILON = 1
    EPSILON = 1
    MIN_EPSILON = 0.01
    DE_EPSILON = 0.01

    # 新建游戏，新建agent
    env = gym_gvgai.make('gvgai-aai-lvl0-v0')
    print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
    actions = env.env.GVGAI.actions()
    agent = tRL(actions, LEARNING_RATE, GAMMA, EPSILON, LAMBDA)

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
        print("reward:%d\tticks:%d\tmean:%f"%(reward, ticks, totalReward/(round+1)))
        agent.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DE_EPSILON*round)

    # 存下agent信息
    agent.to_json(AGENT)

    # 存储结果信息
    pd.DataFrame(result, columns=col).to_csv(RESULT)

