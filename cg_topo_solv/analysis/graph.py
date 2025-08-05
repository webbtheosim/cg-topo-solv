import networkx as nx
import numpy as np

def get_desc(G):
    """
    Compute various network measures for a given networkx graph.

    Args:
        G (networkx.Graph): The input graph for which network measures will be computed.

    Returns:
        numpy.ndarray: An array containing the following network measures in order:
            1. Number of nodes
            2. Number of edges
            3. Algebraic connectivity
            4. Diameter
            5. Radius
            6. Average degree
            7. Average neighbor degree
            8. Network density
            9. Mean degree centrality
            10. Mean betweenness centrality
            11. Degree assortativity coefficient
    """
    x1 = nx.number_of_nodes(G)
    x2 = nx.number_of_edges(G)
    x3 = nx.algebraic_connectivity(G)
    x4 = nx.diameter(G)
    x5 = nx.radius(G)
    degrees = [degree for _, degree in G.degree()]
    x6 = sum(degrees) / len(G.nodes())
    x7 = np.mean(list(nx.average_neighbor_degree(G).values()))
    x8 = nx.density(G)
    x9 = np.mean(list(nx.degree_centrality(G).values()))
    x10 = np.mean(list(nx.betweenness_centrality(G).values()))
    x11 = nx.degree_assortativity_coefficient(G)

    return np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])


def coarsen_graph(G):
    """Coarsen a graph while preserving topology."""
    def is_ring(G):
        return all(d == 2 for _, d in G.degree()) and nx.is_connected(G) and len(G) >= 3

    preserved = {n for n, d in G.degree() if d != 2}
    H = nx.Graph()
    visited = set()
    new_node_id = 0

    if is_ring(G):
        nodes = list(G.nodes)
        k = len(nodes)
        m = k // 2
        ring_nodes = [f"cg_{i}" for i in range(m)]
        H.add_nodes_from(ring_nodes)
        for i in range(m):
            H.add_edge(ring_nodes[i], ring_nodes[(i + 1) % m])
        return H

    H.add_nodes_from(preserved)

    for u in preserved:
        for v in list(G.neighbors(u)):
            if v in preserved:
                if u < v:
                    H.add_edge(u, v)
            elif v not in visited:
                path = [u, v]
                visited.add(v)
                prev, current = u, v
                while True:
                    next_nodes = [n for n in G.neighbors(current) if n != prev]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    prev, current = current, next_node
                    visited.add(current)
                    path.append(current)
                    if current in preserved:
                        break

                k = len(path) - 2
                m = k // 2
                if m > 0:
                    new_chain = [f"cg_{new_node_id + i}" for i in range(m)]
                    new_node_id += m
                    H.add_nodes_from(new_chain)
                    H.add_edge(path[0], new_chain[0])
                    for i in range(m - 1):
                        H.add_edge(new_chain[i], new_chain[i + 1])
                    H.add_edge(new_chain[-1], path[-1])
                else:
                    if path[0] != path[-1]:
                        H.add_edge(path[0], path[-1])

    return H