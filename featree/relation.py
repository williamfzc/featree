import networkx as nx
import requests
import tqdm

URL_BASE = "http://127.0.0.1:9411"
URL_FILE_LIST = f"{URL_BASE}/file/list"
URL_FILE_RELATION = f"{URL_BASE}/file/relation"


def gen_graph() -> nx.Graph:
    files = requests.get(URL_FILE_LIST).json()

    # Initialize the graph
    g = nx.Graph()

    # Add nodes
    for file in files:
        g.add_node(file)

    # Prepare to track min and max scores for normalization
    min_score = float('inf')
    max_score = float('-inf')

    # First pass: find min and max scores
    relations_dict = {}
    for each_file in tqdm.tqdm(files):
        relations = requests.get(URL_FILE_RELATION, params={"path": each_file}).json()
        relations_dict[each_file] = relations
        for each_relation in relations:
            score = each_relation["score"]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score

    # Second pass: add edges with normalized weights
    for each_file, relations in relations_dict.items():
        for each_relation in relations:
            normalized_score = (each_relation["score"] - min_score) / (max_score - min_score)
            if normalized_score > 0:  # Only add edges with positive weights
                g.add_edge(each_file, each_relation["name"], weight=normalized_score)

    return g
