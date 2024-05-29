import typing

import networkx as nx
import numpy as np
import pandas as pd
import requests
import tqdm
import treelib
from community import community_louvain
from langchain_core.language_models import BaseLLM
from pydantic import BaseModel
from treelib import Tree, Node
from langchain_community.llms import Ollama
from loguru import logger

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


class FeatreeNode(BaseModel):
    nid: str
    desc: str = ""
    files: typing.List[str] = []
    children: typing.List["FeatreeNode"] = []


class Featree(object):
    ROOT = "0"

    def __init__(self, data: treelib.Tree):
        self._data: treelib.Tree = data
        self._desc_dict = dict()
        self._summary = ""

    def leaves(self):
        return [each for each in self._data.leaves() if len(each.data) > 1]

    def infer_leaves(self, llm: BaseLLM):
        for node in tqdm.tqdm(self.leaves()):
            self.infer_node(llm, node)

    def infer_node(self, llm: BaseLLM, node: Node):
        prompt = f"""
You are a master software developer.

Please help summarize the potential business functions contained in the following list of files. 
Return a single sentence that encapsulates the business functions of all the files combined. 
Do not include any file paths, only provide the combined one-sentence summary.
Do not include any prefixes and wrappers.

<content>
{"\n".join(node.data)}
</content>

Here is your Summary:
        """
        desc = llm.invoke(prompt)
        self._desc_dict[node.identifier] = desc

    def infer_summary(self, llm: BaseLLM):
        prompt = f"""
You are a master software developer.

Please help me summarize the functionalities provided by the code repository.
I will provide summaries of some modules. Please summarize these summaries and provide a concise overall summary.

<content>
{"\n".join(self._desc_dict.values())}
</content>

Here is your Summary:
"""
        self._summary = llm.invoke(prompt)

    def to_node_tree(self, node_id: str = None) -> FeatreeNode:
        if not node_id:
            node_id = self._data.root
        node = FeatreeNode(
            nid=node_id,
            desc=self._desc_dict.get(node_id, ""),
            files=self._data.get_node(node_id).data or [],
        )

        if node_id == self._data.root:
            node.desc = self._summary

        for child_node in self._data.children(node_id):
            if child_node.identifier not in self._desc_dict:
                continue
            child_node = self.to_node_tree(child_node.identifier)
            node.children.append(child_node)
        return node


def gen_tree() -> Featree:
    graph = gen_graph()

    # Set the threshold for community size
    threshold = int(0.1 * len(graph.nodes))
    if threshold < 20:
        threshold = 20

    tree = Tree()
    tree.create_node(identifier=Featree.ROOT)

    # Generate the final communities
    recursive_community_detection(graph, threshold, tree, tree.root)
    ret = Featree(tree)
    logger.info(f"leaves: {len(ret.leaves())}")

    llm = Ollama(model="llama3")
    ret.infer_leaves(llm)
    ret.infer_summary(llm)
    return ret
