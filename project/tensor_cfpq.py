import networkx as nx
import pyformlang
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from pyformlang.rsa import RecursiveAutomaton
from scipy import sparse
from typing import Dict, Set
from collections import defaultdict

from project.adj_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.graph_utils import graph_to_nfa


def msbfs(
    inter_nfa: AdjacencyMatrixFA,
    graph_nfa: AdjacencyMatrixFA,
    rsm_nfa: AdjacencyMatrixFA,
) -> Dict[str, Set[tuple[int, int]]]:
    num_inter_states = inter_nfa.num_states

    comb_trans_matrix = sparse.csr_matrix(
        (num_inter_states, num_inter_states), dtype=bool
    )
    for mat in inter_nfa.boolean_decomposition.values():
        comb_trans_matrix += mat
    comb_trans_matrix = comb_trans_matrix.transpose()

    idx_to_state = inter_nfa.idx_to_states

    inter_start_idxs = set()
    inter_final_idxs = set()
    for idx, st in idx_to_state.items():
        _, (nonterm, rsm_state) = st.value
        rsm_state_obj = State((nonterm, rsm_state))
        rsm_idx = rsm_nfa.states_to_idx[rsm_state_obj]
        if rsm_idx in rsm_nfa.start_states_indices:
            inter_start_idxs.add(idx)
        if rsm_idx in rsm_nfa.final_states_indices:
            inter_final_idxs.add(idx)

    inter_start_idxs_list = list(inter_start_idxs)
    num_starts = len(inter_start_idxs_list)

    blocks = []
    for start_idx in inter_start_idxs_list:
        block = sparse.csr_matrix((num_inter_states, 1), dtype=bool)
        block[start_idx, 0] = True
        blocks.append(block)

    front = sparse.vstack(blocks)
    visited = front.copy()

    while front.count_nonzero() > 0:
        new_front_parts = []
        for i in range(num_starts):
            slice_start = i * num_inter_states
            slice_end = (i + 1) * num_inter_states
            new_slice = comb_trans_matrix @ front[slice_start:slice_end]
            new_front_parts.append(new_slice)
        new_front = sparse.vstack(new_front_parts)
        front = (new_front > visited).astype(bool)
        visited += front

    reachable_pairs_by_nonterm: Dict[str, Set[tuple[int, int]]] = defaultdict(set)

    for i, start_idx in enumerate(inter_start_idxs_list):
        for final_idx in inter_final_idxs:
            if visited[i * num_inter_states + final_idx, 0]:
                start_graph_state, (start_nonterm, _) = idx_to_state[start_idx].value
                final_graph_state, (final_nonterm, _) = idx_to_state[final_idx].value

                if start_nonterm != final_nonterm:
                    continue

                reachable_pairs_by_nonterm[str(start_nonterm)].add(
                    (
                        graph_nfa.states_to_idx[start_graph_state],
                        graph_nfa.states_to_idx[final_graph_state],
                    )
                )

    return dict(reachable_pairs_by_nonterm)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[tuple[int, int]]:
    rsm_nfa = NondeterministicFiniteAutomaton()
    for nonterm, box in rsm.boxes.items():
        for u, v, label in box.dfa.to_networkx().edges(data="label"):
            rsm_nfa.add_transition((nonterm, u), label, (nonterm, v))
        for s in box.dfa.start_states:
            rsm_nfa.add_start_state((nonterm, s))
        for f in box.dfa.final_states:
            rsm_nfa.add_final_state((nonterm, f))

    rsm_nfa_mat = AdjacencyMatrixFA(rsm_nfa)
    graph_nfa_mat = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    for nonterm in rsm.boxes:
        key = str(nonterm)
        if key not in graph_nfa_mat.boolean_decomposition:
            graph_nfa_mat.boolean_decomposition[key] = sparse.csr_matrix(
                (graph_nfa_mat.num_states, graph_nfa_mat.num_states), dtype=bool
            )

    changed = True
    while changed:
        changed = False

        intersection = intersect_automata(graph_nfa_mat, rsm_nfa_mat)

        new_reachable_pairs_by_nonterm = msbfs(intersection, graph_nfa_mat, rsm_nfa_mat)

        for nonterm_str, pairs in new_reachable_pairs_by_nonterm.items():
            mat = graph_nfa_mat.boolean_decomposition[nonterm_str]
            for u, v in pairs:
                if not mat[u, v]:
                    mat[u, v] = True
                    changed = True

    result = set()
    start_sym_str = str(rsm.initial_label)
    if start_sym_str in graph_nfa_mat.boolean_decomposition:
        start_mat = graph_nfa_mat.boolean_decomposition[start_sym_str]
        for u_idx, v_idx in zip(*start_mat.nonzero()):
            if (
                u_idx in graph_nfa_mat.start_states_indices
                and v_idx in graph_nfa_mat.final_states_indices
            ):
                result.add(
                    (
                        graph_nfa_mat.idx_to_states[u_idx].value,
                        graph_nfa_mat.idx_to_states[v_idx].value,
                    )
                )

    return result


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
