import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def configuration_model(k):
    if sum(k) % 2 != 0:
        raise ValueError("sum(k) % 2 != 0")

    # every node u gets k[u] stubs 
    stubs = []
    for u in range(len(k)):
        for i in range(k[u]):
            stubs.append(u)

    # random stubs - shuffle and take consecutive pairs
    np.random.shuffle(stubs)
    adj_list = []
    for i in range(0, len(stubs), 2):
        adj_list.append((stubs[i], stubs[i + 1]))
    return adj_list

def is_simple(adj_list):
    seen = set()
    for u, v in adj_list:
        if u == v:
            return False  # self-loop
        if (u, v) in seen or (v, u) in seen:
            return False  # multi-edge
        seen.add((u, v))
    return True

def draw_multigraph(adj_list, n):
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(adj_list)

    pos = nx.spring_layout(G, seed=1, k=0.5)
    plt.figure(figsize=(4, 4))
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos)

    for u, v, key in G.edges(keys=True):
        n_edges = G.number_of_edges(u, v)
        if u == v:
            rad = 0.3 + 0.15 * key
        else:
            rad = 0.2 * (key - (n_edges - 1) / 2)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v, key)],
            connectionstyle=f"arc3,rad={rad}",
            arrows=True,
            arrowstyle="-",
            min_source_margin=10,
            min_target_margin=10,
        )
    plt.axis("off")
    plt.show()
    return G

def success_rate(k, tries=1000):
    counter = 0
    for i in range(tries):
        adj_list = configuration_model(k)   
        if is_simple(adj_list):
            counter += 1
    return counter/tries