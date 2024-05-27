import networkx as nx
import numpy as np
import pandas as pd
import requests
from community import community_louvain

URL_BASE = "http://127.0.0.1:9411"
URL_FILE_LIST = f"{URL_BASE}/file/list"
URL_FILE_RELATION = f"{URL_BASE}/file/relation"

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


def recursive_community_detection(G, threshold, community_id=0):
    # Initial community detection on the whole graph or subgraph
    partition = community_louvain.best_partition(G, weight="weight")

    # Create a dictionary to hold the final communities
    final_communities = {}

    # Step 2: Check each community
    for comm in set(partition.values()):
        # Get the nodes in this community
        community_nodes = [
            nodes for nodes in partition.keys() if partition[nodes] == comm
        ]

        if len(community_nodes) > threshold:
            # Step 3: Further split the large community
            subgraph = G.subgraph(community_nodes)
            sub_communities = recursive_community_detection(
                subgraph, threshold, community_id
            )

            # Merge sub-communities into final communities dict
            for sub_comm, sub_nodes in sub_communities.items():
                final_communities[sub_comm] = sub_nodes
                community_id = max(community_id, sub_comm + 1)
        else:
            # If community size is within threshold, add directly to final communities
            final_communities[community_id] = community_nodes
            community_id += 1

    return final_communities


# Set the threshold for community size
threshold = int(0.02 * len(files))
if threshold < 10:
    threshold = 10
print(f"threshold: {threshold}")

# Generate the final communities
final_communities = recursive_community_detection(G, threshold)

# Print the final communities
print(final_communities)
