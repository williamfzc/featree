import csv
import os
import typing
from collections import deque, OrderedDict, Counter

import networkx
import networkx as nx
import numpy as np
import pandas as pd
import tqdm
import treelib
from community import community_louvain
from loguru import logger
from pydantic import BaseModel
from treelib import Tree, Node

from featree.config import GenTreeConfig
from featree.llm import LLM, get_llm, get_mock_llm
from featree.relation import gen_graph


def postorder_traversal(tree: Tree, node_id: str, visit: typing.Callable):
    node = tree.get_node(node_id)
    if node is not None:
        for child in tree.children(node_id):
            postorder_traversal(tree, child.identifier, visit)
        visit(node)


def bfs_traversal(tree: Tree, start_node_id: str, visit: typing.Callable):
    queue = deque([start_node_id])

    while queue:
        node_id = queue.popleft()
        node = tree.get_node(node_id)
        visit(node)
        for child in tree.children(node_id):
            queue.append(child.identifier)


def dfs_traversal(tree: Tree, node_id: str, visit: typing.Callable):
    node = tree.get_node(node_id)
    if node is not None:
        visit(node)
        for child in tree.children(node_id):
            dfs_traversal(tree, child.identifier, visit)


def weighted_graph_density(G):
    total_weight = sum(weight for u, v, weight in G.edges.data("weight", default=1))
    max_possible_weight = (
        max(weight for u, v, weight in G.edges.data("weight", default=1))
        * (len(G) - 1)
        * len(G)
        / 2
    )
    return total_weight / max_possible_weight


def graph_connected_count(parent_g: nx.Graph, g1: nx.Graph, g2: nx.Graph) -> int:
    count = 0
    for u in g1.nodes:
        for v in g2.nodes:
            if parent_g.has_edge(u, v):
                count += parent_g.edges[u, v]["weight"]
    return count


class SymbolTable(object):
    def __init__(self):
        self.files = []
        self.file_index_dict = dict()

    def at(self, src: str, dst: str) -> typing.List[str]:
        key = (src, dst)
        if key not in self.file_index_dict:
            return []
        return self.file_index_dict[key]


def load_symbol_table(f: str) -> SymbolTable:
    with open(f, "r") as file:
        files = pd.read_csv(file, nrows=0).columns.tolist()[1:]

    ret = SymbolTable()
    ret.files = files

    with open(f, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        next(reader)
        index_dict = dict()
        for src_file_index, row in enumerate(reader):
            valid_data = row[1:]
            for dst_file_index, val in enumerate(valid_data):
                if not val:
                    continue
                index_dict[(files[src_file_index], files[dst_file_index])] = val.split(
                    "|"
                )
    ret.file_index_dict = index_dict
    return ret


class Cluster(BaseModel):
    files: typing.List[str] = []
    symbols: Counter[str] = Counter()
    leader_file: str = ""
    hub_score: float = 0.0


def louvain(g, **kwargs):
    partition = community_louvain.best_partition(g, **kwargs)
    keys = sorted(partition.keys())
    counter = OrderedDict()
    for k in keys:
        comm = partition[k]
        if comm not in counter:
            counter[comm] = []
        counter[comm].append(k)
    return counter


def recursive_community_detection(
    g: nx.Graph,
    leaves_limit: int,
    density_ratio: float,
    tree: treelib.Tree,
    parent: treelib.Node,
):
    # Initial community detection on the whole graph or subgraph
    part_dict = {each_node: index for index, each_node in enumerate(sorted(g.nodes()))}
    counter = louvain(g, partition=part_dict, random_state=42)

    for comm, community_nodes in counter.items():
        # Step 3: Further split the large community
        cur_community_graph = g.subgraph(community_nodes)
        # dead loop
        if cur_community_graph.order() == g.order():
            continue

        # try splitting this graph
        for community_components in nx.connected_components(cur_community_graph):
            component_graph = cur_community_graph.subgraph(community_components).copy()
            # todo: inVs and outVs
            node_data = Cluster(files=community_components)
            n = tree.create_node(
                parent=parent,
                data=node_data,
            )

            # too small, stop
            if len(component_graph) < leaves_limit:
                continue

            density = weighted_graph_density(component_graph)
            if density < density_ratio:
                recursive_community_detection(
                    component_graph, leaves_limit, density_ratio, tree, n
                )


class FeatreeNode(BaseModel):
    nid: str
    desc: str = ""
    cluster: Cluster = None
    children: typing.List["FeatreeNode"] = []

    @property
    def files(self):
        return self.cluster.files


class _TreeBase(object):
    ROOT = "0"

    def __init__(
        self,
        data: treelib.Tree,
        digraph: nx.DiGraph,
        graph: nx.Graph,
        config: GenTreeConfig,
    ):
        self._data: treelib.Tree = data
        self._desc_dict = dict()
        self._leave_di_graph = nx.DiGraph()
        self._origin_di_graph = digraph
        self._origin_graph = graph
        self.config = config

    def leaves(self) -> typing.List[Node]:
        return [
            each
            for each in self._data.leaves()
            if each.data and len(each.data.files) > 1
        ]

    def desc(self, node: Node) -> str:
        return self._desc_dict[node.identifier]

    def walk_postorder(self, visit_cb: typing.Callable, node_id: str = None):
        if not node_id:
            node_id = self._data.root
        postorder_traversal(self._data, node_id, visit_cb)

    def walk_dfs(self, visit_cb: typing.Callable, node_id: str = None):
        if not node_id:
            node_id = self._data.root
        dfs_traversal(self._data, node_id, visit_cb)

    def walk_bfs(self, visit_cb: typing.Callable, node_id: str = None):
        if not node_id:
            node_id = self._data.root
        bfs_traversal(self._data, node_id, visit_cb)

    def build_graph(self):
        # 1. add nodes to graph
        g = nx.DiGraph()
        self._leave_di_graph = g
        leaves = set([each.identifier for each in self.leaves()])
        g.add_nodes_from(leaves)

        # 2. link these nodes by branches
        def _walk(n: Node):
            if n.identifier in leaves:
                return

            def _link(nn: Node):
                if nn.identifier in leaves:
                    return
                children = self._data.children(nn.identifier)
                for child1 in children:
                    for child2 in children:
                        ci1, ci2 = child1.identifier, child2.identifier
                        if not ((ci1 in leaves) and (ci2 in leaves)):
                            continue
                        if ci1 == ci2:
                            continue

                        if not g.has_edge(ci1, ci2):
                            weight = 1
                            for u in child1.data.files:
                                for v in child2.data.files:
                                    if self._origin_graph.has_edge(u, v):
                                        weight += 1
                            g.add_edge(ci1, ci2, weight=weight)

            self.walk_bfs(_link, n.identifier)

        # TODO
        # self.walk_postorder(_walk, self.ROOT)

    def calc_leader_files(self):
        leaders = []
        ranks = networkx.pagerank(self._origin_di_graph)
        for each_cluster_id in self._leave_di_graph.nodes():
            each_cluster = self._data.get_node(each_cluster_id)
            each_files = each_cluster.data.files
            max_file = None
            max_score = float("-inf")

            for each_file in each_files:
                score = ranks[each_file]
                if score > max_score:
                    max_score = score
                    max_file = each_file

            each_cluster.data.leader_file = max_file
            leaders.append((each_cluster_id, max_file, max_score))

        edge_count = {}
        for each_cluster_id, each_file, each_rank in sorted(leaders):
            each_neighbor_files = list(self._origin_di_graph.neighbors(each_file))

            for each_target_cluster_id in self._leave_di_graph.nodes():
                each_cluster = self._data.get_node(each_target_cluster_id)
                for each_neighbor_file in each_neighbor_files:
                    if each_neighbor_file not in each_cluster.data.files:
                        continue
                    if each_cluster_id == each_target_cluster_id:
                        continue

                    if each_cluster.data.symbols and len(each_cluster.data.symbols) > 0:
                        most_common_symbol = each_cluster.data.symbols.most_common(1)[
                            0
                        ][1]
                    else:
                        most_common_symbol = 0

                    edge = (each_cluster_id, each_target_cluster_id)
                    if edge in edge_count:
                        edge_count[edge] += most_common_symbol
                    else:
                        edge_count[edge] = most_common_symbol

        edge_by_cluster = {}
        for edge, count in edge_count.items():
            each_cluster_id, each_target_cluster_id = edge
            if each_cluster_id not in edge_by_cluster:
                edge_by_cluster[each_cluster_id] = Counter()
            edge_by_cluster[each_cluster_id][each_target_cluster_id] += count

        for each_cluster_id, target_clusters in edge_by_cluster.items():
            most_common_edges = target_clusters.most_common(10)
            for each_target_cluster_id, count in most_common_edges:
                self._leave_di_graph.add_edge(
                    each_cluster_id, each_target_cluster_id, weight=count
                )

        # Authority ~= Function
        # Hub ~= Danger files
        hubs, authorities = nx.hits(self._leave_di_graph)
        hub_scores = np.array(list(hubs.values()))
        median_score = np.median(hub_scores)

        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)

        for node, hub_score in top_hubs:
            node_data = self._data.get_node(node).data
            node_data.hub_score = hub_score

            if hub_score < median_score:
                self._leave_di_graph.remove_edges_from(list(self._leave_di_graph.edges(node)))

    def load_symbols_to_graph(self):
        if not self.config.include_symbols:
            return

        symbol_table = load_symbol_table(self.config.symbol_csv_file)
        leaves = self.leaves()
        for src_leaf in leaves:
            each_counter = Counter()
            for dst_leaf in leaves:
                for each_src_file in src_leaf.data.files:
                    for each_dst_file in dst_leaf.data.files:
                        symbols = symbol_table.at(each_src_file, each_dst_file)
                        for each in symbols:
                            each_counter[each] += 1
            src_leaf.data.symbols.update(each_counter)

    def neighbors(self, node: Node, dis_limit: float = None) -> typing.List[Node]:
        neighbors = self._leave_di_graph.neighbors(node.identifier)
        # check these neighbors
        nodes = [self._data.get_node(each) for each in neighbors]
        ret = []

        src_leader_file_dir = os.path.dirname(node.data.leader_file)

        start_graph = self._origin_graph.subgraph(node.data.files)
        for each_node in nodes:
            # check leader file
            if src_leader_file_dir == os.path.dirname(each_node.data.leader_file):
                continue

            if not dis_limit:
                ret.append(each_node)
                continue

            end_graph = self._origin_graph.subgraph(each_node.data.files)
            dis = nx.graph_edit_distance(start_graph, end_graph)
            if dis < dis_limit:
                ret.append(each_node)

        return ret


class Featree(_TreeBase):
    def infer_leaves(self, llm: LLM):
        for node in tqdm.tqdm(self.leaves()):
            self.infer_node(llm, node)

    def infer_branches(self, llm: LLM):
        def inner(node: Node):
            if node.identifier in self._desc_dict:
                return

            # infer from children's desc
            desc_list = []
            nid = node.identifier
            for each_child in self._data.children(nid):
                each_child_nid = each_child.identifier
                if each_child_nid in self._desc_dict:
                    desc = self._desc_dict[each_child_nid]
                    desc_list.append(desc)
            summary = self._infer_summary(llm, "\n".join(desc_list))
            self._desc_dict[nid] = summary

        postorder_traversal(self._data, self.ROOT, inner)

    def infer_node(self, llm: LLM, node: Node):
        content = "\n".join(node.data.files)
        desc = self._infer_summary(llm, content)
        self._desc_dict[node.identifier] = desc

    def _infer_summary(self, llm: LLM, prompt: str) -> str:
        prompt = f"""
<task>
Please help summarize the potential functions contained in the following content. 
Return a brief sentence that encapsulates the functions of all the files combined. 
</task>

<content>
{prompt}
</content>

<requirement>
NO ANY PREFIXES!
</requirement>
            """
        desc = llm.ask(prompt)
        return desc

    def to_node_tree(self, node_id: str = None) -> FeatreeNode:
        if not node_id:
            node_id = self._data.root
        node = FeatreeNode(
            nid=node_id,
            desc=self._desc_dict.get(node_id, ""),
            cluster=self._data.get_node(node_id).data or Cluster(),
        )

        for child_node in self._data.children(node_id):
            child_node = self.to_node_tree(child_node.identifier)
            node.children.append(child_node)
        return node


def gen_tree(config: GenTreeConfig = None) -> Featree:
    if not config:
        config = GenTreeConfig()

    digraph, graph = gen_graph(config)
    sub_graphs = [
        graph.subgraph(component).copy() for component in nx.connected_components(graph)
    ]
    logger.info("relation graph ready")

    # Set the threshold for community size
    leaves_limit = int(config.leaves_limit_ratio * len(graph.nodes))
    if leaves_limit < config.leaves_limit:
        leaves_limit = config.leaves_limit

    tree = Tree()
    tree.create_node(identifier=Featree.ROOT, data=Cluster(files=[]))

    for each_sub_graph in sub_graphs:
        recursive_community_detection(
            each_sub_graph, leaves_limit, config.density_ratio, tree, tree.root
        )

    ret = Featree(tree, digraph, graph, config)
    logger.info("tree ready")

    if config.infer:
        llm = get_llm()
    else:
        llm = get_mock_llm()

    ret.infer_leaves(llm)
    ret.infer_branches(llm)

    # build graph onto these nodes
    ret.build_graph()
    logger.info("build graph ready")

    ret.load_symbols_to_graph()
    ret.calc_leader_files()

    return ret
