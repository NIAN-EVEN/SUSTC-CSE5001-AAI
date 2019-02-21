#!/usr/bin/env python
import gym
import gym_gvgai
import Agent as Agent

env = gym_gvgai.make('gvgai-aai-lvl0-v0')
agent = Agent.Agent()
print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))

result = []
ticks = 0
reward = 0
totalReward = 0
for r in range(100):
    reward = 0
    ticks = 0

    # reset environment
    stateObs = env.reset()
    actions = env.env.GVGAI.actions()
    for t in range(2000):
        # choose action based on trained policy
        env.render()
        action_id = agent.act(t, stateObs, actions)
        # do action and get new state and its reward
        stateObs, increScore, done, debug = env.step(action_id)
        # print("Action " + str(action_id) + " tick " + str(t+1) + " reward " + str(increScore) + " win " + debug["winner"])
        # break loop when terminal state is reached
        if done:
            if debug["winner"] == "USER_WIN":
                print("WIN!!!")
            break

        ticks += 1
        reward += increScore
    totalReward += reward
    print("reward:%d\tticks:%d\tmean:%f"%(reward, ticks, totalReward/(r+1)))
    result.append([reward, ticks])
