import os
import re
import subprocess

import networkx as nx
import pandas
from pandas import DataFrame
from loguru import logger

from featree.config import GenTreeConfig


def gen_graph(
    config: GenTreeConfig,
) -> (nx.Graph, DataFrame):
    csv_file = config.csv_file
    symbol_csv_file = config.symbol_csv_file
    if os.path.exists(csv_file):
        os.remove(csv_file)
    if os.path.exists(symbol_csv_file):
        os.remove(symbol_csv_file)

    commands = [
        config.gossiphs_bin,
        "relation",
        "--project-path",
        config.project_path,
        "--csv",
        csv_file,
        "--strict",
    ]
    if config.include_symbols:
        commands.extend(["--symbol-csv", symbol_csv_file])
    if config.exclude_regex:
        commands.extend(["--exclude-file-regex", config.exclude_regex])
    logger.info(f"gossiphs cmd: {commands}")
    subprocess.check_call(commands)

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

    for row_number, (i, row) in enumerate(df.iterrows()):
        for j, value in enumerate(row):
            if value > 0:
                cur_file = df.columns[j]
                if g.has_node(i) and g.has_node(cur_file):
                    g.add_edge(i, cur_file, weight=value)

    isolated_nodes = [node for node in g.nodes() if g.degree(node) == 0]
    g.remove_nodes_from(isolated_nodes)
    return g
