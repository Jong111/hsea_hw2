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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-type', type=str, help='graph type', default='regular')
    parser.add_argument('--n-nodes', type=int, help='the number of nodes', default=11)  # 可见一共有2的50000次方个解（默认情况下）
    parser.add_argument('--n-d', type=int, help='the number of degrees for each node', default=10)
    parser.add_argument('--T', type=int, help='the number of fitness evaluations', default=10)
    parser.add_argument('--seed-g', type=int, help='the seed of generating regular graph', default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--gset-id', type=int, default=1)
    parser.add_argument('--sigma', type=float, help='hyper-parameter of mutation operator', default=.1)
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
    print(g1)
    g2 = np.where(ind == 1)[0]
    return nx.cut_size(graph, g1, g2) / n_edges


# 挑选一部分个体
def selection(pop, k, graph, n_edges, scores):
    # scores = [score(i, graph, n_edges) for i in pop]
    # print(scores)
    idx = np.random.randint(len(pop))
    for i in np.random.randint(0, len(pop), k - 1):
        if scores[i] > scores[idx]:
            idx = i
    return pop[idx]


def get_best_fitness(pop, graph, n_edges):
    best_fitness = -1
    for i in range(len(pop)):
        if score(pop[i], graph, n_edges) > best_fitness:
            best_fitness = score(pop[i], graph, n_edges)
    return best_fitness


def get_fitness(graph, x, n_edges, threshold=0):
    # 这里我将-1修改为了0
    x_eval = np.where(x >= threshold, 1, -1)
    # print("x_eval ", x_eval)
    # 获得Cuts值需要将图分为两部分, 这里默认以0为阈值把解分成两块.
    # g1和g2中的元素是结点的下标
    g1 = np.where(x_eval == -1)[0]
    # print("g1 ",g1)
    g2 = np.where(x_eval == 1)[0]
    return [nx.cut_size(graph, g1, g2) / n_edges, g1]  # cut_size返回的是连接图的两部分g1,g2的桥的权重之


def main(args=get_args()):  # 优化目标是找到一组图的划分，使得cut_size最大
    print(args)
    yy1 = []
    xx1 = []
    graph, n_nodes, n_edges = graph_generator(args)

    tmppop = generate_population(n_nodes, 100)
    pop = tmppop2pop(tmppop)
    for i in pop:
        print(i)
    scores = [score(i, graph, n_edges) for i in pop]
    # selected = selection(pop, 3, graph, n_edges, scores)
    parents = [selection(pop, 3, graph, n_edges, scores) for _ in range(len(pop))]
    for i in parents:
        print(i)
    print(len(parents))
    print(get_best_fitness(pop, graph, n_edges))
    # print("selected ",  selected)
    # parents = [selection(pop, 3, graph, n_edges) for _ in range(len(pop))]
    # print(get_best_fitness(pop, graph, n_edges))
    np.random.seed(args.seed)  # 为下面调用的random.rand设置一个随机数种子
    # 返回一个1*n_nodes大小的由浮点数组成的随机数组
    # x在算法中扮演什么角色？ x在将图划分为两部分的过程中起辅助划分的作用
    # x是一个list，因此每个个体的表示形式也是一个list
    x = np.random.rand(n_nodes)  # 这里x使用实数值表示, 也可以直接使用01串表示, 并使用01串上的交叉变异算子，n_nodes是图中结点的数量
    # 在原始种群中只有1个个体，因此原始parent也只有一个（不需要刻意选择）
    best_fitness = get_fitness(graph, x, n_edges)[0]
    print(best_fitness)
    # for i in range(args.T):  # 简单的(1+1)ES
    #     print(i)
    #     tmp = x + np.random.randn(n_nodes) * args.sigma
    #     tmp_fitness = get_fitness(graph, tmp, n_edges)
    #     if tmp_fitness > best_fitness:
    #         x, best_fitness = tmp, tmp_fitness
    #         yy1.append(best_fitness)
    #         xx1.append(i)
    #         print(i, best_fitness)
    # # 调用演化算法
    # evolutionary_algorithm(n_nodes, args.sigma, args.T, graph, n_edges)
    # # 画图
    # plt.plot(xx1, yy1, label="Result")
    # plt.plot(xx2, yy2, label="Result2")
    # plt.xlabel("times")
    # plt.ylabel("cut_size")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
