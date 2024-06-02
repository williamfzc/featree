import os
import subprocess

import networkx as nx
import pandas

from featree.config import GenTreeConfig


def gen_graph(config: GenTreeConfig) -> nx.Graph:
    csv_file = "featree-temp.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)

    subprocess.check_call(
        [
            config.gossiphs_bin,
            "relation",
            "--project-path",
            config.project_path,
            "--csv",
            csv_file,
            "--strict",
        ]
    )

    df = pandas.read_csv(csv_file, index_col=0)

    g = nx.Graph()
    for column in df.columns:
        g.add_node(column)

    for i, row in df.iterrows():
        for j, value in enumerate(row):
            if value > 0:
                g.add_edge(i, df.columns[j], weight=value)

    return g
