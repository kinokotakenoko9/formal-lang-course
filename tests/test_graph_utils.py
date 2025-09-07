from pathlib import Path
import networkx as nx

from project.graph_utils import (
    GraphInfo,
    _get_graph_info_by_graph,
    get_graph_info_by_name,
    create_and_save_2_cycle_labeled_graph,
)


def test__get_graph_info_by_graph():
    g = nx.MultiDiGraph()
    g.add_edge(0, 1, label="a")
    g.add_edge(1, 2, label="b")
    g.add_edge(2, 0, label="a")

    info = _get_graph_info_by_graph(g)

    assert info["num_vertices"] == 3
    assert info["num_edges"] == 3
    assert info["labels"] == {"a", "b"}
    assert (
        GraphInfo(
            num_vertices=3,
            num_edges=3,
            labels={"a", "b"},
        )
        == info
    )


def test_get_graph_info_by_name():
    actual = get_graph_info_by_name("pizza")
    expected = GraphInfo(
        num_vertices=671,
        num_edges=1980,
        labels={
            "disjointWith",
            "type",
            "subClassOf",
            "onProperty",
            "first",
            "rest",
            "someValuesFrom",
            "label",
            "allValuesFrom",
            "comment",
            "unionOf",
            "equivalentClass",
            "intersectionOf",
            "range",
            "domain",
            "hasValue",
            "distinctMembers",
            "subPropertyOf",
            "inverseOf",
            "complementOf",
            "versionInfo",
            "minCardinality",
            "oneOf",
        },
    )
    assert actual == expected


def test_create_and_save_2_cycle_labeled_graph(tmp_path: Path):
    filename = "test_graph_1.dot"

    tmp_output_file = tmp_path / filename
    create_and_save_2_cycle_labeled_graph(
        n1=2, n2=3, label1="Σ", label2="Λ", path=str(tmp_output_file)
    )
    actual = nx.nx_pydot.read_dot(tmp_output_file)

    expected = nx.nx_pydot.read_dot(Path(__file__).parent / "test_graphs" / filename)

    assert tmp_output_file.exists()
    assert tmp_output_file.read_text() != ""
    assert nx.utils.graphs_equal(actual, expected)
