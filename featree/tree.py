import typing
from collections import deque, OrderedDict

import networkx as nx
import tqdm
import treelib
from community import community_louvain
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


def recursive_community_detection(
    g: nx.Graph,
    leaves_limit: int,
    density_ratio: float,
    tree: treelib.Tree,
    parent: treelib.Node,
):
    # Initial community detection on the whole graph or subgraph
    part_dict = {each_node: index for index, each_node in enumerate(sorted(g.nodes()))}
    partition = community_louvain.best_partition(
        g, partition=part_dict, random_state=42
    )

    # Step 2: Check each community
    keys = sorted(partition.keys())
    counter = OrderedDict()
    for k in keys:
        comm = partition[k]
        if comm not in counter:
            counter[comm] = []
        counter[comm].append(k)

    for comm, community_nodes in counter.items():
        # Step 3: Further split the large community
        cur_community_graph = g.subgraph(community_nodes)
        # dead loop
        if cur_community_graph.order() == g.order():
            continue

        # try splitting this graph
        for community_components in nx.connected_components(cur_community_graph):
            component_graph = cur_community_graph.subgraph(community_components).copy()
            n = tree.create_node(parent=parent, data=community_components)

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
    files: typing.List[str] = []
    children: typing.List["FeatreeNode"] = []


class _TreeBase(object):
    ROOT = "0"

    def __init__(self, data: treelib.Tree):
        self._data: treelib.Tree = data
        self._desc_dict = dict()

    def leaves(self):
        return [each for each in self._data.leaves() if len(each.data) > 1]

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
        content = "\n".join(node.data)
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
            files=self._data.get_node(node_id).data or [],
        )

        for child_node in self._data.children(node_id):
            child_node = self.to_node_tree(child_node.identifier)
            node.children.append(child_node)
        return node


def gen_tree(config: GenTreeConfig = None) -> Featree:
    if not config:
        config = GenTreeConfig()

    graph = gen_graph(config)
    sub_graphs = [
        graph.subgraph(component).copy() for component in nx.connected_components(graph)
    ]

    # Set the threshold for community size
    leaves_limit = int(config.leaves_limit_ratio * len(graph.nodes))
    if leaves_limit < config.leaves_limit:
        leaves_limit = config.leaves_limit

    tree = Tree()
    tree.create_node(identifier=Featree.ROOT, data=set())

    for each_sub_graph in sub_graphs:
        recursive_community_detection(
            each_sub_graph, leaves_limit, config.density_ratio, tree, tree.root
        )

    ret = Featree(tree)

    if config.infer:
        llm = get_llm()
    else:
        llm = get_mock_llm()

    ret.infer_leaves(llm)
    ret.infer_branches(llm)

    return ret
