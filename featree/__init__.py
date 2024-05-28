import networkx as nx
import numpy as np
import pandas as pd
import requests
from community import community_louvain
from treelib import Tree

URL_BASE = "http://127.0.0.1:9411"
URL_FILE_LIST = f"{URL_BASE}/file/list"
URL_FILE_RELATION = f"{URL_BASE}/file/relation"


def gen_graph():
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


def recursive_community_detection(G, threshold, community_id, tree):
    # Initial community detection on the whole graph or subgraph
    # file -> comm dict
    partition = community_louvain.best_partition(G)

    # Create a dictionary to hold the final communities
    final_communities = {}

    # Step 2: Check each community
    for comm in set(partition.values()):
        # Get the nodes in this community
        community_nodes = [nodes for nodes in partition.keys() if partition[nodes] == comm]

        if len(community_nodes) > threshold:
            # Step 3: Further split the large community
            subgraph = G.subgraph(community_nodes)

            cur = community_id[0]
            tree.create_node(cur, parent=tree.root)

            sub_communities = recursive_community_detection(subgraph, threshold, community_id, tree)

            assert not set(sub_communities.keys()) & set(final_communities.keys())
            # Merge sub-communities into final communities dict
            final_communities.update(sub_communities)
            # sub_communities belong to community_id
            for each_comm in sub_communities:
                tree.create_node(each_comm, parent=cur)
        else:
            # If community size is within threshold, add directly to final communities
            final_communities[community_id[0]] = community_nodes
            community_id[0] += 1

    return final_communities


if __name__ == "__main__":
    graph = gen_graph()

    # Set the threshold for community size
    threshold = int(0.1 * len(graph.nodes))
    if threshold < 20:
        threshold = 20
    print(f"threshold: {threshold}")

    tree = Tree()

    # Generate the final communities
    final_communities = recursive_community_detection(graph, threshold, [0], tree)

    for k, v in final_communities.items():
        print(f"{k}: {len(v)}")
