from dataclasses import dataclass
from typing import Set, Tuple, Optional
from collections import defaultdict, deque

import networkx as nx
from pyformlang.finite_automaton import Symbol, State
from pyformlang.rsa import RecursiveAutomaton


@dataclass(frozen=True)
class RSMState:
    nonterm: Symbol
    state: State


@dataclass(frozen=True)
class GSSNode:
    rsm_st: RSMState
    graph_node: int


@dataclass(frozen=True)
class Config:
    rsm_st: RSMState
    graph_node: int
    gss_node: GSSNode


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Optional[Set[int]] = None,
    final_nodes: Optional[Set[int]] = None,
) -> Set[Tuple[int, int]]:
    start_nodes = start_nodes if start_nodes is not None else set(graph.nodes)
    final_nodes = final_nodes if final_nodes is not None else set(graph.nodes)

    graph_adj = defaultdict(lambda: defaultdict(set))
    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            graph_adj[u][Symbol(label)].add(v)

    rsm_boxes = {}
    for label in set(rsm.labels) | {rsm.initial_label}:
        box = rsm.get_box(label)
        if box is not None:
            dfa = box.dfa
            trans = defaultdict(lambda: defaultdict(set))

            for from_state, by_symbol in dfa._transition_function._transitions.items():
                for sym, to_state in by_symbol.items():
                    if isinstance(to_state, set):
                        for s in to_state:
                            trans[from_state][sym].add(s)
                    else:
                        trans[from_state][sym].add(to_state)

            rsm_boxes[label] = {
                "transitions": trans,
                "start_states": {dfa.start_state} if dfa.start_state else set(),
                "final_states": set(dfa.final_states),
            }

    gss_nodes = {}
    gss_edges = defaultdict(list)

    dummy_root = GSSNode(RSMState(Symbol("$"), State(0)), -1)
    gss_nodes[dummy_root] = None

    queue = deque()
    seen: Set[Config] = set()
    result_pairs = set()

    initial_box = rsm_boxes[rsm.initial_label]
    for start_state in initial_box["start_states"]:
        for start_node in start_nodes:
            rsm_st = RSMState(rsm.initial_label, start_state)

            gss_node = GSSNode(rsm_st, start_node)
            gss_nodes.setdefault(gss_node, None)

            gss_edges[gss_node].append((dummy_root, rsm_st))

            config = Config(rsm_st, start_node, gss_node)
            if config not in seen:
                queue.append(config)
                seen.add(config)

    while queue:
        conf = queue.popleft()

        box = rsm_boxes.get(conf.rsm_st.nonterm)

        transitions = box["transitions"]
        cur_state = conf.rsm_st.state

        # process terminal transitions
        if cur_state in transitions:
            for symbol, next_states in transitions[cur_state].items():
                if symbol not in rsm_boxes and symbol in graph_adj[conf.graph_node]:
                    for neighbor in graph_adj[conf.graph_node][symbol]:
                        for next_rsm_st in next_states:
                            new_rsm_st = RSMState(conf.rsm_st.nonterm, next_rsm_st)
                            new_conf = Config(new_rsm_st, neighbor, conf.gss_node)
                            if new_conf not in seen:
                                queue.append(new_conf)
                                seen.add(new_conf)

        # process nonterminal calls
        if cur_state in transitions:
            for symbol, next_states in transitions[cur_state].items():
                if symbol in rsm_boxes:
                    called_box = rsm_boxes[symbol]
                    for call_start_state in called_box["start_states"]:
                        call_rsm_st = RSMState(symbol, call_start_state)
                        call_gss_node = GSSNode(call_rsm_st, conf.graph_node)

                        pop_set = gss_nodes.get(call_gss_node)
                        if pop_set is not None:
                            for ret_state in next_states:
                                gss_edges[call_gss_node].append(
                                    (
                                        conf.gss_node,
                                        RSMState(conf.rsm_st.nonterm, ret_state),
                                    )
                                )
                                for ret_node in pop_set:
                                    return_conf = Config(
                                        RSMState(conf.rsm_st.nonterm, ret_state),
                                        ret_node,
                                        conf.gss_node,
                                    )
                                    if return_conf not in seen:
                                        queue.append(return_conf)
                                        seen.add(return_conf)
                            continue

                        gss_nodes[call_gss_node] = None
                        for ret_state in next_states:
                            gss_edges[call_gss_node].append(
                                (
                                    conf.gss_node,
                                    RSMState(conf.rsm_st.nonterm, ret_state),
                                )
                            )

                        entry_conf = Config(call_rsm_st, conf.graph_node, call_gss_node)
                        if entry_conf not in seen:
                            queue.append(entry_conf)
                            seen.add(entry_conf)

        # proccess if cur state final
        if cur_state in box["final_states"]:
            if gss_nodes[conf.gss_node] is None:
                gss_nodes[conf.gss_node] = set()
            gss_nodes[conf.gss_node].add(conf.graph_node)

            for caller, return_rsm_st in gss_edges.get(conf.gss_node, []):
                if caller is dummy_root:
                    result_pairs.add((conf.gss_node.graph_node, conf.graph_node))
                else:
                    return_conf = Config(return_rsm_st, conf.graph_node, caller)
                    if return_conf not in seen:
                        queue.append(return_conf)
                        seen.add(return_conf)

    return {(u, v) for u, v in result_pairs if u in start_nodes and v in final_nodes}
