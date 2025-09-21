from typing import Dict, Iterable
from networkx import MultiDiGraph
from scipy.sparse import dok_matrix, csr_matrix, kron, identity
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)

from project.graph_utils import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        if nfa is None:
            self.num_states = 0
            self.start_states_indices = set()
            self.final_states_indices = set()
            self.states_to_idx = {}
            self.idx_to_states = {}
            self.boolean_decomposition = {}
            return

        self.start_states = nfa.start_states
        self.final_states = nfa.final_states

        self.num_states = len(nfa.states)
        states = sorted(list(nfa.states), key=lambda s: s.value)
        # map states to 0..<num_states
        self.states_to_idx = {s: i for i, s in enumerate(states)}
        self.idx_to_states = {i: s for i, s in enumerate(states)}
        self.start_states_indices = {self.states_to_idx[s] for s in nfa.start_states}
        self.final_states_indices = {self.states_to_idx[s] for s in nfa.final_states}

        self.boolean_decomposition = self.get_boolean_decomposition(nfa)

    def get_boolean_decomposition(
        self, nfa: NondeterministicFiniteAutomaton
    ) -> Dict[Symbol, csr_matrix]:
        boolean_matrices_dok = {}

        for u, symbol, v in nfa:
            # add new symbols
            if symbol.value not in boolean_matrices_dok:
                boolean_matrices_dok[symbol.value] = dok_matrix(
                    (self.num_states, self.num_states), dtype=bool
                )

            # mark transition
            boolean_matrices_dok[symbol.value][
                self.states_to_idx[u], self.states_to_idx[v]
            ] = True

        # convert dok matrices to csr matrices
        return {s: m.tocsr() for s, m in boolean_matrices_dok.items()}

    def _get_transitive_closure(self) -> csr_matrix:
        if self.num_states == 0:
            return csr_matrix((0, 0), dtype=bool)

        adj_matrix = sum(
            self.boolean_decomposition.values(),
            csr_matrix((self.num_states, self.num_states), dtype=bool),
        )

        adj_matrix += identity(self.num_states, dtype=bool, format="csr")

        prev_nnz = -1
        while adj_matrix.nnz != prev_nnz:
            prev_nnz = adj_matrix.nnz
            adj_matrix += adj_matrix @ adj_matrix

        return adj_matrix

    def accepts(self, word: Iterable[Symbol]) -> bool:
        if not self.start_states_indices:
            return False

        current_states_vec = csr_matrix((1, self.num_states), dtype=bool)
        for idx in self.start_states_indices:
            current_states_vec[0, idx] = True

        for s in word:
            if s not in self.boolean_decomposition:
                return False

            # update current states with one step
            current_states_vec = current_states_vec @ self.boolean_decomposition[s]

        # check if any of the final states are in the set of current states
        final_states_mask = csr_matrix((self.num_states, 1), dtype=bool)
        for i in self.final_states_indices:
            final_states_mask[i, 0] = True

        return (current_states_vec @ final_states_mask).nnz > 0

    def is_empty(self) -> bool:
        if not self.start_states_indices or not self.final_states_indices:
            return True

        transitive_closure = self._get_transitive_closure()

        # check for a path between any start and any final state.
        for start_idx in self.start_states_indices:
            for final_idx in self.final_states_indices:
                if transitive_closure[start_idx, final_idx]:
                    return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    # start with empty fa
    intersection = AdjacencyMatrixFA(None)

    intersection.num_states = automaton1.num_states * automaton2.num_states

    # interate over common symbol and build transitions
    for s in (
        automaton1.boolean_decomposition.keys()
        & automaton2.boolean_decomposition.keys()
    ):
        intersection.boolean_decomposition[s] = kron(
            automaton1.boolean_decomposition[s],
            automaton2.boolean_decomposition[s],
            format="csr",
        )

    # calculate start and final states
    for s1_idx, s1 in automaton1.idx_to_states.items():
        for s2_idx, s2 in automaton2.idx_to_states.items():
            new_idx = s1_idx * automaton2.num_states + s2_idx

            new_state = State((s1.value, s2.value))
            intersection.states_to_idx[new_state] = new_idx
            intersection.idx_to_states[new_idx] = new_state

            if (
                s1_idx in automaton1.start_states_indices
                and s2_idx in automaton2.start_states_indices
            ):
                intersection.start_states_indices.add(new_idx)

            if (
                s1_idx in automaton1.final_states_indices
                and s2_idx in automaton2.final_states_indices
            ):
                intersection.final_states_indices.add(new_idx)

    return intersection


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    if not graph.nodes or not regex:
        return set()

    graph_nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_dfa = AdjacencyMatrixFA(regex_to_dfa(regex))

    intersection = intersect_automata(graph_nfa, regex_dfa)

    if intersection.num_states == 0:
        return set()

    transitive_closure = intersection._get_transitive_closure()

    result = set()
    num_regex_states = regex_dfa.num_states

    # iterate through all (start state, final satate) pairs
    for start_idx in intersection.start_states_indices:
        for final_idx in intersection.final_states_indices:
            # if a path exists decode the indices back to the graph nodes
            if transitive_closure[start_idx, final_idx]:
                graph_start_node_idx = start_idx // num_regex_states
                graph_final_node_idx = final_idx // num_regex_states

                start_node = graph_nfa.idx_to_states[graph_start_node_idx].value
                final_node = graph_nfa.idx_to_states[graph_final_node_idx].value

                result.add((start_node, final_node))

    return result
