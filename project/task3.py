from typing import Dict, Iterable
from networkx import MultiDiGraph
from scipy.sparse import dok_matrix
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)


class AdjacencyMatrixFA:
    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        self.start_states = nfa.start_states
        self.final_states = nfa.final_states

        self.boolean_decomposition = self.get_boolean_decomposition(nfa)

    def get_boolean_decomposition(
        self, nfa: NondeterministicFiniteAutomaton
    ) -> Dict[Symbol, dok_matrix]:
        boolean_matrices_dok = {}
        num_states = len(nfa.states)
        # map states to 0..<num_states
        states = sorted(list(nfa.states), key=lambda s: s.value)
        states_to_idx = {s: i for i, s in enumerate(states)}

        for u, l, v in nfa:
            # add new symbols
            if l not in boolean_matrices_dok:
                boolean_matrices_dok[l.value] = dok_matrix(
                    (num_states, num_states), dtype=bool
                )

            # mark transition
            boolean_matrices_dok[l.value][states_to_idx[u], states_to_idx[v]] = True

        # convert dok matrices to csr matrices
        return {s: m.tocsr() for s, m in boolean_matrices_dok.items()}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        pass

    def is_empty(self) -> bool:
        pass


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    pass


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    pass
