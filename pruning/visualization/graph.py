import networkx as nx


def build_2L_graph(N):
    """
    Build attention visualization graph with N tokens per layers.

    Parameters
    ----------
    N : int
        Number of tokens per layer.
    """
    G = nx.Graph()

    G.add_nodes_from([f"0{i}" for i in range(N)])
    G.add_nodes_from([f"1{i}" for i in range(N)])
    G.add_node("2")

    # Add edges between nodes in different sets
    for i in range(N):
        for j in range(N):
            G.add_edge(f"0{i}", f"1{j}")
        G.add_edge(f"1{i}", "2")

    pos = {}
    for i in range(N):
        pos[f"0{i}"] = (i, 0)
        pos[f"1{i}"] = (i, -1)
    pos["2"] = (N - 1, -2)

    return G, pos


def build_1L_graph(N):
    """
    Build attention visualization graph with N tokens per layers.

    Parameters
    ----------
    N : int
        Number of tokens per layer.
    """
    G = nx.Graph()

    G.add_nodes_from([f"0{i}" for i in range(N)])
    G.add_node("1")

    # Add edges between nodes in different sets
    for i in range(N):
        G.add_edge(f"0{i}", "1")

    pos = {}
    for i in range(N):
        pos[f"0{i}"] = (i, 0)
    pos["1"] = (N / 2, -1)

    return G, pos
