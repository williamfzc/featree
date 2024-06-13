import os
import re
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

    # exclude rule
    exclude_regex = None
    if config.exclude_regex:
        exclude_regex = re.compile(config.exclude_regex)

    g = nx.Graph()
    for column in df.columns:
        if exclude_regex and exclude_regex.search(column):
            continue
        g.add_node(column)

    for i, row in df.iterrows():
        for j, value in enumerate(row):
            if value > 0:
                if g.has_node(i) and g.has_node(df.columns[j]):
                    g.add_edge(i, df.columns[j], weight=value)

    isolated_nodes = [node for node in g.nodes() if g.degree(node) == 0]
    g.remove_nodes_from(isolated_nodes)
    return g
