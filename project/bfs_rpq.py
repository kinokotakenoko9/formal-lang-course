from networkx import MultiDiGraph

from project.adj_matrix_fa import AdjacencyMatrixFA
from project.graph_utils import graph_to_nfa, regex_to_dfa
from scipy.sparse import dok_matrix, csr_matrix


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_mfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_mfa = AdjacencyMatrixFA(regex_to_dfa(regex))

    regex_start_idx = list(regex_mfa.start_states_indices)[0]

    common_alphabet = (
        graph_mfa.boolean_decomposition.keys() & regex_mfa.boolean_decomposition.keys()
    )

    graph_start_list = sorted(list(graph_mfa.start_states_indices))
    k = len(graph_start_list)

    fronts = []
    reachable = []

    for g_start_idx in graph_start_list:
        front = dok_matrix((graph_mfa.num_states, regex_mfa.num_states), dtype=bool)

        front[g_start_idx, regex_start_idx] = True

        fronts.append(front.tocsr())
        reachable.append(front.copy().tocsr())

    graph_matrices_t = {
        symbol: matrix.transpose().tocsr()
        for symbol, matrix in graph_mfa.boolean_decomposition.items()
        if symbol in common_alphabet
    }

    while any(f.nnz > 0 for f in fronts):
        for i in range(k):
            if fronts[i].nnz == 0:
                continue

            new_front = csr_matrix(fronts[i].shape, dtype=bool)

            for symbol in common_alphabet:
                g_mat_t = graph_matrices_t[symbol]
                r_mat = regex_mfa.boolean_decomposition[symbol]

                new_front += (g_mat_t @ fronts[i]) @ r_mat

            fronts[i] = new_front > reachable[i]

            reachable[i] += fronts[i]

    result = set()

    g_final_indices = graph_mfa.final_states_indices
    r_final_indices = regex_mfa.final_states_indices

    for i, g_start_idx in enumerate(graph_start_list):
        r_matrix = reachable[i]

        for r_final_idx in r_final_indices:
            col = r_matrix[:, r_final_idx]

            reached_g_indices = col.nonzero()[0]

            for g_final_idx in reached_g_indices:
                if g_final_idx in g_final_indices:
                    start_node_val = graph_mfa.idx_to_states[g_start_idx].value
                    final_node_val = graph_mfa.idx_to_states[g_final_idx].value

                    result.add((start_node_val, final_node_val))

    return result
