import networkx as nx
import pyformlang
from pyformlang.cfg import CFG, Variable, Terminal, Production, Epsilon
from collections import deque


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    nullable_vars = cfg.get_nullable_symbols()
    cnf_cfg = cfg.to_normal_form()
    new_productions = set(cnf_cfg.productions)

    for var in nullable_vars:
        new_productions.add(Production(Variable(var.value), [Epsilon()]))

    wcnf_cfg = CFG(start_symbol=cnf_cfg.start_symbol, productions=new_productions)

    return wcnf_cfg.remove_useless_symbols()


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    wcnf = cfg_to_weak_normal_form(cfg)

    eps_prods = []
    term_prods = {}
    var_prods = {}

    for p in wcnf.productions:
        head = p.head
        body = p.body

        if not body or (len(body) == 1 and isinstance(body[0], Epsilon)):
            eps_prods.append(head)

        elif len(body) == 1 and isinstance(body[0], Terminal):
            t_val = body[0].value
            if t_val not in term_prods:
                term_prods[t_val] = []
            term_prods[t_val].append(head)

        elif len(body) == 2:
            left = body[0]
            right = body[1]
            if (left, right) not in var_prods:
                var_prods[(left, right)] = []
            var_prods[(left, right)].append(head)

    r_set = set()
    queue = deque()

    for node in graph.nodes:
        for var in eps_prods:
            triple = (var, node, node)
            if triple not in r_set:
                r_set.add(triple)
                queue.append(triple)

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label in term_prods:
            for var in term_prods[label]:
                triple = (var, u, v)
                if triple not in r_set:
                    r_set.add(triple)
                    queue.append(triple)

    while queue:
        var_i, u, v = queue.popleft()

        current_r = list(r_set)

        for var_j, u_prime, v_prime in current_r:
            if v_prime == u:
                pair = (var_j, var_i)
                if pair in var_prods:
                    for head in var_prods[pair]:
                        new_triple = (head, u_prime, v)
                        if new_triple not in r_set:
                            r_set.add(new_triple)
                            queue.append(new_triple)

            if v == u_prime:
                pair = (var_i, var_j)
                if pair in var_prods:
                    for head in var_prods[pair]:
                        new_triple = (head, u, v_prime)
                        if new_triple not in r_set:
                            r_set.add(new_triple)
                            queue.append(new_triple)

    result_pairs = set()
    start_symbol = wcnf.start_symbol

    for var, u, v in r_set:
        if var == start_symbol:
            if u in start_nodes and v in final_nodes:
                result_pairs.add((u, v))

    return result_pairs
