import networkx as nx
import numpy as np
import pandas as pd
import requests

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

    g = nx.Graph()
    for file in files:
        g.add_node(file)

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if df_normal.iloc[i, j] > 0:
                g.add_edge(files[i], files[j], weight=df_normal.iloc[i, j])
    return g
