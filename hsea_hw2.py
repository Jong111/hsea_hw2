import networkx as nx
import numpy as np
import argparse
import matplotlib.pyplot as plt
import ioh


def generate_regular_graph(args):
    # 这里简单以正则图为例, 鼓励同学们尝试在其他类型的图(具体可查看如下的nx文档)上测试算法性能
    # nx文档 https://networkx.org/documentation/stable/reference/generators.html
    graph = nx.random_graphs.random_regular_graph(d=args.n_d, n=args.n_nodes, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)


def generate_gset_graph(args):
    # 这里提供了比较流行的图集合: Gset, 用于进行分割
    dir = './Gset/'
    fname = dir + 'G' + str(args.gset_id) + '.txt'
    graph_file = open(fname)
    n_nodes, n_e = graph_file.readline().rstrip().split(' ')
    print(n_nodes, n_e)
    nodes = [i for i in range(int(n_nodes))]
    edges = []
    for line in graph_file:
        n1, n2, w = line.split(' ')
        edges.append((int(n1) - 1, int(n2) - 1, int(w)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return graph, len(graph.nodes), len(graph.edges)


def graph_generator(args):
    if args.graph_type == 'regular':
        return generate_regular_graph(args)
    elif args.graph_type == 'gset':
        return generate_gset_graph(args)
    else:
        raise NotImplementedError(f'Wrong graph_tpye')


# 在原始的框架代码中每次的划分方式为：将满足一定条件的节点和不满足该条件的结点分开
# 但每次x的选取是随机的，因此x中哪些元素满足该条件也是随机的，即每次结点的划分也是随机的
def get_fitness(graph, x, n_edges, threshold=0):
    # 这里我将-1修改为了0
    x_eval = np.where(x >= threshold, 1, -1)
    # print("x_eval ", x_eval)
    # 获得Cuts值需要将图分为两部分, 这里默认以0为阈值把解分成两块.
    # g1和g2中的元素是结点的下标
    g1 = np.where(x_eval == -1)[0]
    # print("g1 ",g1)
    g2 = np.where(x_eval == 1)[0]
    return nx.cut_size(graph, g1, g2) / n_edges  # cut_size返回的是连接图的两部分g1,g2的桥的权重之和


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-type', type=str, help='graph type', default='regular')
    parser.add_argument('--n-nodes', type=int, help='the number of nodes', default=50)  # 可见一共有2的50000次方个解（默认情况下）
    parser.add_argument('--n-d', type=int, help='the number of degrees for each node', default=10)
    parser.add_argument('--T', type=int, help='the number of fitness evaluations', default=500)
    parser.add_argument('--seed-g', type=int, help='the seed of generating regular graph', default=50)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--gset-id', type=int, default=1)
    parser.add_argument('--sigma', type=float, help='hyper-parameter of mutation operator', default=.3)
    args = parser.parse_known_args()[0]
    return args


# 产生种群
def generate_population(n_nodes, n_pop):
    tmppop = [np.random.uniform(-1, 1, n_nodes).tolist() for _ in range(n_pop)]
    return tmppop


# 将实数值表示的个体转换为01串表示的个体
def tmppop2pop(tmppop):
    pop = []
    for i in range(len(tmppop)):
        tmp = []
        for j in range(len(tmppop[i])):
            if tmppop[i][j] >= 0:
                tmp.append(1)
            else:
                tmp.append(0)
        pop.append(tmp)
    return pop


# 为每个个体打分
def score(individual, graph, n_edges):
    ind = np.array(individual)
    g1 = np.where(ind == 0)[0]
    # print(g1)
    g2 = np.where(ind == 1)[0]
    return nx.cut_size(graph, g1, g2) / n_edges


# 挑选一部分个体
def selection(pop, k, graph, n_edges, scores):
    # scores = [score(i, graph, n_edges) for i in pop]
    idx = np.random.randint(len(pop))
    for i in np.random.randint(0, len(pop), k - 1):
        if scores[i] > scores[idx]:
            idx = i
    return pop[idx]


# 选出父母（方法比较朴素）
def generate_parents(pop, k, graph, n_edges, scores):
    return [selection(pop, k, graph, n_edges, scores) for _ in range(len(pop) // 5)]


# 交叉
def crossover(p1, p2):
    c1, c2 = p1.copy(), p2.copy()
    crossoverPos1 = np.random.randint(1, len(c1) - 1)
    # crossoverPos2 = np.random.randint(1, len(c1) - 1)
    # c1 = p1[:crossoverPos1] + p2[crossoverPos1: crossoverPos2] + p1[crossoverPos2:]
    # c2 = p2[:crossoverPos1] + p1[crossoverPos1: crossoverPos2] + p2[crossoverPos2:]
    c1 = p1[:crossoverPos1] + p2[crossoverPos1:]
    c2 = p2[:crossoverPos1] + p1[crossoverPos1:]
    return c1, c2


# 变异
def mutation(individual, m_rate):
    for i in range(len(individual)):
        if np.random.rand() < m_rate:
            individual[i] = 1 - individual[i]


# 计算两个个体之间的距离（用汉明距离表示）
def dis(ind1, ind2):
    res = 0
    for i in range(len(ind1)):
        res += abs(ind1[i]-ind2[i])
    return res


# # 计算种群中任意两个个体之间的汉明距离的平均值
# def calAverageDistance(pop):
#     res = 0
#     for i in range(len(pop)):
#         for j in range(i+1, len(pop)):
#             res += dis(pop[i], pop[j])
#     return res


# 选取一对父母产生孩子（这里选取哪两个元素作为父母的策略可以改进）
# 并且我这里省略了select_survivor的步骤，我的Children直接是由parents替换形成的
# 严格来说，Cildren = select_survivor(offspring 并 parents)
def children(pop, m_rate, k, graph, n_edges, scores):
    Children = []
    parents = generate_parents(pop, k, graph, n_edges, scores)
    # res = (calAverageDistance(pop) * 2)/(len(pop) * (len(pop) - 1))
    for i in range(0, len(parents)-1, 2):
        while 1:
            p1idx = np.random.randint(len(parents))
            p2idx = np.random.randint(len(parents))
            if p1idx != p2idx:
                break
        p1, p2 = parents[p1idx], parents[p2idx]
        c1, c2 = crossover(p1, p2)
        mutation(c1, m_rate)
        mutation(c2, m_rate)
        # print(len(p1), ' ', len(p2), ' ', len(c1), ' ', len(c2))
        # p1和c2比较像，p2和c1比较像
        if dis(p1, c1) + dis(p2, c2) > dis(p1, c2) + dis(p2, c1):
            if score(p1, graph, n_edges) > score(c2, graph, n_edges):
                c2 = p1
            if score(p2, graph, n_edges) > score(c1, graph, n_edges):
                c1 = p2
        else:
            if score(p1, graph, n_edges) > score(c1, graph, n_edges):
                c1 = p1
            if score(p2, graph, n_edges) > score(c2, graph, n_edges):
                c2 = p2
        Children.append(c1)
        Children.append(c2)
    # 下面我来写一个简单的select_survivor策略
    next_pop = pop + Children
    offspring = []
    next_scores = [[score(i, graph, n_edges), i] for i in next_pop]
    tmpNext_scores = sorted(next_scores, key=lambda Score: Score[0], reverse=True)
    for i in range(len(pop)):
        offspring.append(tmpNext_scores[i][1])
    return offspring


# 返回种群中的最优解的fitness
def get_best_fitness(pop, graph, n_edges):
    best_fitness = -1
    for i in range(len(pop)):
        if score(pop[i], graph, n_edges) > best_fitness:
            best_fitness = score(pop[i], graph, n_edges)
    return best_fitness


# 演化
def evolutionary_algorithm(n_nodes, n_pop, m_rate, times, k, graph, n_edges):
    # 随机初始化一个种群
    tmppop = generate_population(n_nodes, n_pop)
    xx2 = []
    yy2 = []
    pop = tmppop2pop(tmppop)
    for generation in range(times):
        scores = [score(i, graph, n_edges) for i in pop]
        Children = children(pop, m_rate, k, graph, n_edges, scores)
        pop = Children
        xx2.append(generation)
        yy2.append(get_best_fitness(pop, graph, n_edges))
        print(generation, get_best_fitness(pop, graph, n_edges))
    return xx2, yy2


# 每个解是一个个体，
def main(args=get_args()):  # 优化目标是找到一组图的划分，使得cut_size最大
    print(args)
    yy1 = []
    xx1 = []
    graph, n_nodes, n_edges = graph_generator(args)
    np.random.seed(args.seed)  # 为下面调用的random.rand设置一个随机数种子
    # 返回一个1*n_nodes大小的由浮点数组成的随机数组
    # x在算法中扮演什么角色？ x在将图划分为两部分的过程中起辅助划分的作用
    # x是一个list，因此每个个体的表示形式也是一个list
    x = np.random.rand(n_nodes)  # 这里x使用实数值表示, 也可以直接使用01串表示, 并使用01串上的交叉变异算子，n_nodes是图中结点的数量
    # 在原始种群中只有1个个体，因此原始parent也只有一个（不需要刻意选择）
    best_fitness = get_fitness(graph, x, n_edges)
    for i in range(args.T):  # 简单的(1+1)ES
        print(i)
        tmp = x + np.random.randn(n_nodes) * args.sigma
        tmp_fitness = get_fitness(graph, tmp, n_edges)
        if tmp_fitness > best_fitness:
            x, best_fitness = tmp, tmp_fitness
            yy1.append(best_fitness)
            xx1.append(i)
            print(i, best_fitness)
    # 调用演化算法
    xx2, yy2 = evolutionary_algorithm(n_nodes, 500, args.sigma, args.T, args.k, graph, n_edges)
    # 画图
    plt.plot(xx1, yy1, label="Result")
    plt.plot(xx2, yy2, label="Result2")
    plt.xlabel("times")
    plt.ylabel("cut_size")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
