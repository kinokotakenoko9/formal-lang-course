from cfpq_data import download, graph_from_csv
from cfpq_data.graphs.generators import labeled_two_cycles_graph
from networkx import MultiDiGraph
from typing import TypedDict
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot

class GraphInfo(TypedDict):
    num_vertices: int
    num_edges: int
    labels: set

def _get_graph_info_by_graph(graph: MultiDiGraph) -> GraphInfo:
    return GraphInfo(
        num_vertices=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        labels=set(nx.get_edge_attributes(graph, "label").values())
    )

def _get_graph_by_name(name: str) -> MultiDiGraph:
    graph_path = download(name)
    graph = graph_from_csv(graph_path)

    return graph

def get_graph_info_by_name(name: str) -> GraphInfo:
    graph = _get_graph_by_name(name)

    return _get_graph_info_by_graph(graph)

def create_and_save_2_cycle_labeled_graph(n1: int, n2: int, label1: str, label2: str, path: str) -> None:
    graph = labeled_two_cycles_graph(n1, n2, labels=(label1, label2))

    pydot_graph = to_pydot(graph)
    pydot_graph.write_raw(path)
