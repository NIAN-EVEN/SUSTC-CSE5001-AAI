# -*-coding:utf-8 -*-

from solver11849058 import *
import time

def test_ANN():
    print('7')
    generate_DATA(7)
    ann = ANN()
    ann.input_num = 7
    ann.node_num = 12
    ann.hidden_out_num = ann.node_num - ann.input_num
    ann.connection = np.array([[True,True,True,True],
                                [True,True,True,True],
                                [False,True,True,True],
                                [True,True,True,True],
                                [True,True,True,True],
                                [True,True,True,True],
                                [True,True,True,True],
                                [True,True,True,True],
                                [False,True,True,False],
                                [False,False,True,False],
                                [False,False,False,True],
                                [False,False,False,False]])
    ann.weight = np.array([[-42.8,-75.3,-85.0,59.9],
                            [-32.4,-32.0,-28.1,12.9],
                            [0,43.2,28.6,-13.5],
                            [-5.6,-41.1,-28.0,13.0],
                            [-23.6,-34.5,-28.0,13.0],
                            [-33.6,-34.8,-28.2,13.0],
                            [-33.6,-34.8,-28.2,13.0],
                            [41.6,39.8,29.3,-13.4],
                            [0,-58.9,-47.6,0],
                            [0,0,-41.3,0],
                            [0,0,0,81.8],
                            [0,0,0,0]])
    ann.evaluate()
    # self.MBP(100)
    ann.output_structure()
    print('8')
    generate_DATA(8)
    ann.input_num = 8
    ann.node_num = 13
    ann.hidden_out_num = ann.node_num - ann.input_num
    ann.connection = np.array([[True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [False, True, True, True],
                                [False, False, True, True],
                                [False, False, False, True],
                                [False, False, False, False]])
    ann.weight = np.array([[-12.4, -40.4, -48.1, 45.7],
                            [25.2, 19.6, 16.0, -10.0],
                            [27.7, 18.9, 16.1, -11.0],
                            [-29.4, -18.1, -15.9, 10.0],
                            [-28.9, -19.1, -16.3, 9.9],
                            [-29.7, -18.5, -15.8, 9.4],
                            [-25.4, -17.3, -15.9, 10.0],
                            [-28.5, -18.8, -15.8, 9.6],
                            [27.8, 20.4, 16.7, -11.4],
                            [0, -67.6, -55.0, 6.8],
                            [0, 0, -26.7, 2.3],
                            [0, 0, 0, 76.3],
                            [0, 0, 0, 0]])
    ann.evaluate()
    # self.MBP(100)
    ann.test_ANN8()
    ann.output_correct_rate()

def test_BP():
    time1 = time.time()
    ann = ANN()
    ann.evaluate()
    iter = 10000
    while iter>0:
        iter-=200
        ann.adam_backprop(200)
        print(10000-iter,':\t',ann.correct_num,ann.score)
    ann.output_correct_rate()
    time2 = time.time()
    print(time2 - time1)

def test_ABP():
    time1 = time.time()
    pop = setup(100)
    count = 0
    for ann in pop:
        ann.evaluate()
        iter = 10000
        while iter>0:
            iter-=400
            ann.adam_backprop(400)
            if ann.correct_num == 32:
                count += 1
                break
        print(pop.index(ann), ':', ann.correct_num,ann.score)
    print("correct:",count)
    count = 0
    for ann in pop:
        if ann.correct_num == 32:
            count+=1
    print("correct:", count)
    time2 = time.time()
    print(time2 - time1)

def test_SA():
    ann = ANN()
    ann.evaluate()
    ann.adam_backprop()
    ann.output_correct_rate()
    ann.mutation()

def test_running():
    pop = setup(20)

def test_mutation():
    pass

def error_reduce():
    generate_DATA(5)

    ann = ANN()
    ann.evaluate()

def determine_alpha():
    time1 = time.time()
    learn_rate = 0.9
    while learn_rate>=0.01:
        pop = setup(100)
        count = 0
        for ann in pop:
            ann.evaluate()
            iter = 10000
            while iter > 0:
                iter -= 200
                ann.adam_backprop(learn_rate,200)
                if ann.correct_num == 2 ** (ann.input_num - 1):
                    count += 1
                    break
        time2 = time.time()
        print(learn_rate,':',count,' in 100')
        learn_rate -= 0.01
    print(time2 - time1)

def test_all():
    generate_DATA(5)
    # 初始化阶段
    pop = setup(M)
    birds = setbirds()
    # 演化阶段
    generation = 0
    while (running(pop, birds)):  # 未测试
        evolve(pop)
        flyaway(birds)
        birds.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
        pop.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
        generation += 1
    # 输出结果
    print_result(pop[0], birds[0])
    print(generation)
    print('pop',pop[0].node_num)
    pop[0].output_correct_rate()
    print('birds',birds[0].node_num)
    birds[0].output_correct_rate()
    if pop[0].correct_num==32 or birds[0].correct_num == 32:
        return 1
    else:
        return 0

def exp_1_iter():
    start = time.time()
    generate_DATA(5)
    pop = setup(M)
    birds = setbirds()
    # 演化阶段
    generation = 0
    while pop[0].correct_num == 32 and pop[0].node_num <= 9:  # 未测试
        evolve(pop)
        flyaway(birds)
        birds.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
        pop.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
        generation += 1
        if time.time()-start>180:
            break
    if pop[0].correct_num==32 and pop[0].node_num<=9:
        result = pop[0]
    # 烟花没有最优有两种情况，1，随机最优，2到时间了
    elif birds[0].correct_num==32 and birds[0].node_num<=9:
        result = birds[0]
    # 到时间了，经验上说演化会更好
    else:
        result = pop[0]
    result.calculate_links()
    return result.node_num, result.link ,generation, result.score

def exp_1():
    start = time.time()
    sum_nodes, sum_links, sum_gener, sum_error = 0, 0, 0, 0
    max_nodes, max_links, max_gener, max_error = 0, 0, 0, 0
    min_nodes, min_links, min_gener, min_error = 100, 100, 100, 100

    for i in range(100):
        nodes, links, gener, error = exp_1_iter()
        sum_nodes += nodes
        if nodes > max_nodes:
            max_nodes = nodes
        if nodes < min_nodes:
            min_nodes = nodes
        sum_links += links
        if links > max_links:
            max_links = links
        if links < min_links:
            min_links = links
        sum_gener += gener
        if gener > max_gener:
            max_gener = gener
        if gener < min_gener:
            min_gener = gener
        sum_error += error
        if error > max_error:
            max_error = error
        if error < min_error:
            min_error = error
    print(sum_nodes / 100, '\t', min_nodes, '\t', max_nodes)
    print(sum_links / 100, '\t', min_links, '\t', max_links)
    print(sum_gener / 100, '\t', min_gener, '\t', max_gener)
    print(sum_error / 100, '\t', min_error, '\t', max_error)
    print(time.time()-start)

def exp_3_iter():
    start = time.time()
    generate_DATA(5)
    pop = setup(M)
    # 演化阶段
    generation = 0
    while pop[0].correct_num == 32 and pop[0].node_num <= 9:  # 未测试
        evolve(pop)
        pop.sort(key=lambda x: (32 - x.correct_num, x.node_num, x.score))
        generation += 1
        if time.time()-start>180:
            break
    result = pop[0]
    result.calculate_links()
    return result.node_num, result.link ,generation, result.score

def exp_3():
    start = time.time()
    sum_nodes, sum_links, sum_gener, sum_error = 0, 0, 0, 0
    max_nodes, max_links, max_gener, max_error = 0, 0, 0, 0
    min_nodes, min_links, min_gener, min_error = 100, 100, 100, 100

    for i in range(100):
        print(i)
        nodes, links, gener, error = exp_3_iter()
        sum_nodes += nodes
        if nodes > max_nodes:
            max_nodes = nodes
        if nodes < min_nodes:
            min_nodes = nodes
        sum_links += links
        if links > max_links:
            max_links = links
        if links < min_links:
            min_links = links
        sum_gener += gener
        if gener > max_gener:
            max_gener = gener
        if gener < min_gener:
            min_gener = gener
        sum_error += error
        if error > max_error:
            max_error = error
        if error < min_error:
            min_error = error
    print(sum_nodes / 100, '\t', min_nodes, '\t', max_nodes)
    print(sum_links / 100, '\t', min_links, '\t', max_links)
    print(sum_gener / 100, '\t', min_gener, '\t', max_gener)
    print(sum_error / 100, '\t', min_error, '\t', max_error)
    print(time.time()-start)



if __name__ =='__main__':
    exp_1()
    # exp_3()





