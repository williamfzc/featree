import networkx as nx
import numpy as np
import pandas as pd
import requests
import treelib
from community import community_louvain
from treelib import Tree

URL_BASE = "http://127.0.0.1:9411"
URL_FILE_LIST = f"{URL_BASE}/file/list"
URL_FILE_RELATION = f"{URL_BASE}/file/relation"


def gen_graph() -> nx.Graph:
    files = requests.get(URL_FILE_LIST).json()
    matrix_size = len(files)

    data = np.zeros((matrix_size, matrix_size), dtype=float)
    df = pd.DataFrame(data=data, columns=files, index=files)

    for each_file in files:
        relations = requests.get(URL_FILE_RELATION, params={"path": each_file}).json()
        for each_relation in relations:
            df.at[each_relation["name"], each_file] += each_relation["score"]

    df_normal: pd.DataFrame = (df - df.min().min()) / (df.max().max() - df.min().min())

    G = nx.Graph()
    for file in files:
        G.add_node(file)

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if df_normal.iloc[i, j] > 0:
                G.add_edge(files[i], files[j], weight=df_normal.iloc[i, j])
    return G


def recursive_community_detection(
    g: nx.Graph, threshold: int, tree: treelib.Tree, parent: treelib.Node
):
    # Initial community detection on the whole graph or subgraph
    # file -> comm dict
    partition = community_louvain.best_partition(g)

    # Step 2: Check each community
    for comm in set(partition.values()):
        # Get the nodes in this community
        community_nodes = [
            nodes for nodes in partition.keys() if partition[nodes] == comm
        ]
        n = tree.create_node(parent=parent, data=community_nodes)

        if len(community_nodes) > threshold:
            # Step 3: Further split the large community
            subgraph = g.subgraph(community_nodes)
            # dead loop
            if subgraph.order() == g.order():
                continue

            recursive_community_detection(subgraph, threshold, tree, n)


class Featree(object):
    def __init__(self, data: treelib.Tree):
        self._data: treelib.Tree = data


def gen_tree() -> Featree:
    graph = gen_graph()

    # Set the threshold for community size
    threshold = int(0.1 * len(graph.nodes))
    if threshold < 20:
        threshold = 20

    tree = Tree()
    tree.create_node(identifier=0)

    # Generate the final communities
    recursive_community_detection(graph, threshold, tree, tree.root)
    return Featree(tree)
