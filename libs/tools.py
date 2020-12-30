import networkx as nx
import numpy as np
from itertools import combinations
from network import Network
from heapq import heappop, heapify


def get_odd_degree_nodes(graph):
    """ Finds all odd degree nodes.

    Args:
         - graph (networkx.Graph): undirected graph

    Returns:
        - (list): list of odd degree nodes
    """
    return [v for v, d in graph.degree() if d % 2 == 1]


def map_osmnx_nodes2integers(graph):
    """ Maps osmnx node ids to integers in range (0, node_count)

    Args:
        - graph (networkx.Graph): graph

    Returns
        - node_count (integer): number of nodes
        - nodes_list (list): list of all nodes converted to integers in range (0, node_count)
    """

    nodes_list = list(graph.nodes)
    nodes_list.sort()
    node_count = len(nodes_list)

    return node_count, nodes_list


def map_osmnx_edges2integers(graph, edges):
    """ Maps edges with osmnx ids to to integers in range (0, node_count)

    Args:
        - graph (networkx.Graph): graph
        - edges (list or networkx.EdgeView): list of osmnx edges

    Returns
        - converted_edges (list): list of all edges with converted nodes to integers in range(0, node_count)
    """

    n, nodes_list = map_osmnx_nodes2integers(graph)

    converted_edges = []
    for v, u in edges:
        # TODO: is .index() the best option, what about using dict()
        converted_edges.append((nodes_list.index(v), nodes_list.index(u)))

    return converted_edges


def create_weighted_complete_graph(pair_weights):
    """ Creates a weighted complete graph with [inverse] negative weights given list of weighted edges.
    (Negative weights are due to using maximal matching algorithm instead of minimal matching algorithm)

    Args:
        - pair_weights (list): list of 3-tuples denoting an edge with weight

    Returns:
        - graph (networkx.Graph): complete graph
    """

    graph = nx.Graph()
    graph.add_weighted_edges_from([(e[0], e[1], - np.round(w, 2)) for e, w in pair_weights.items()])

    return graph


def min_matching(pair_weights):
    """ From weighted edges list forms a complete graph and finds a minimal matching [1]

    Args:
        - pair_weights (list): list of 3-tuples denoting an edge with weight
    Returns:
        - matching_weights (list): list of triples denoting an edge with weight

    Sources:
    .. [1] Galil, Z. (1986). Efficient algorithms for finding maximum matching in graphs. ACM Comput. Surv., 18, 23-38.
    https://www.semanticscholar.org/paper/Efficient-algorithms-for-finding-maximum-matching-Galil/ef1b31b4728615a52e3b8084379a4897b8e526ea?p2df
    """

    complete_graph = create_weighted_complete_graph(pair_weights)
    matching = nx.algorithms.max_weight_matching(complete_graph, True)

    matching_weights = []
    for v, u in matching:
        if (v, u) in pair_weights:
            matching_weights.append((v, u, np.round(pair_weights[v, u], 2)))
        else:
            matching_weights.append((v, u, np.round(pair_weights[u, v], 2)))

    return matching_weights


def get_shortest_distance_for_odd_degrees(graph, odd_degree_nodes):
    """ Finds shortest distance for all odd degree nodes combinations

    Args:
        - graph (networkx.Graph): undirected graph
        - odd_degree_nodes (list): list of odd degree nodes

    Returns:
        - (dict): a dict with shortest distances for each combination of nodes
    """

    pairs = combinations(odd_degree_nodes, 2)

    return {(v, u): nx.dijkstra_path_length(graph, v, u, weight='length') for v, u in pairs}


def get_shortest_paths(graph, nodes):
    """ Creates a list of shortest paths between two [not neighboring] nodes.

    Args:
        - graph (networkx.Graph): undirected graph
        - nodes (list): list of tuples (start, end, weight) for which we find a shortest path

    Return:
        - shortest_paths (list): list of shortest paths between two [not neighboring] nodes
    """

    shortest_paths = []

    for u, v, _ in nodes:
        path = nx.dijkstra_path(graph, u, v)
        shortest_paths.extend([(path[i - 1], path[i]) for i in range(1, len(path))])

    return shortest_paths


def networkx2network(graph):
    """ Converts networkx.Graph() into Network

    Args:
       graph (Graph): a Network object

    Returns:
        A Network object.
    """

    new_graph = nx.Graph()
    nodes = list(graph.nodes())
    nodes.sort()
    edges = [(nodes.index(v), nodes.index(u), np.round(d['length'], 2)) for (v, u, d) in graph.edges(data=True) if 'length' in d]  # noqa
    new_graph.add_weighted_edges_from(edges)

    n = len(nodes)

    return Network(n, edges, directed=False, weighted=True)


def get_double_edge_heap(graph):
    """ Creates a heap for multi-edges sorted [asc] with respect to weight

    Args:
        - graph (networkx.Graph): undirected graph

    Returns:
        - double_edge_heap (heapify dict): heap of multi-edges sorted [asc] with respect to weight
    """

    double_edge_heap = {}

    for v, u, i in graph.edges:
        # There are multiple option for the same edge, not recorded yet
        if i > 0 and (u, v) not in double_edge_heap and (v, u) not in double_edge_heap:
            data = graph.get_edge_data(v, u)
            double_edge_heap[v, u] = []
            for k, d in data.items():
                double_edge_heap[v, u].append((np.round(d['length'], 2), k))

            heapify(double_edge_heap[v, u])

    return double_edge_heap


def convert_path(graph, path, double_edge_heap):
    """ Converts path with respect to multi-edges. It makes sure that each edge in multi-edge set is visited.
    If an edge needs to be visited more than number of multi-edges, the shortest one is always selected.

    Args:
        - graph (networkx.Graph): undirected graph
        - path (list): list of [edges] tuples that form a final path
        - double_edge_heap (heapify dict): heap of multi-edges sorted [asc] with respect to weight

    Returns:
        - path_edge_list (list):
    """

    double_edges = double_edge_heap.keys()
    path_edge_list = []
    perv_node = path.pop()

    def pop_double_edge(deh, e):
        if len(deh[e]) > 1:
            return heappop(deh[e])
        else:
            return deh[e][0]

    while len(path) > 0:
        current_node = path.pop()

        # An edge is a self loop, if self does not exist in the [org] graph then remove from the list
        if current_node == perv_node and not graph.get_edge_data(current_node, perv_node):
            current_node = path.pop()

        edge = (perv_node, current_node)

        # Double edge
        if edge in double_edges:
            _, i = pop_double_edge(double_edge_heap, edge)
        elif edge[::-1] in double_edges:
            _, i = pop_double_edge(double_edge_heap, edge[::-1])
        else:
            i = 0

        path_edge_list.append((perv_node, current_node, i))
        perv_node = current_node

    return path_edge_list


def convert_integer_path2osmnx_nodes(path, osmnx_nodes):
    converted_path = []
    if max(path) > len(osmnx_nodes):
        return converted_path

    osmnx_nodes = list(osmnx_nodes)
    osmnx_nodes.sort()

    for p in path:
        converted_path.append(osmnx_nodes[p])

    return converted_path


def convert_final_path_to_coordinates(org_graph, final_path):
    """ Converts final path of [osmnx] edges into list of (lat, log) tuples which are then read by Leaflet JS library.

    Args:
        - org_graph (networkx.Graph): directed [original] graph
        - final_path (list): list of [edges] tuples that form a final path

    Returns:
        - path (list): list of (lat, log) tuples
    """

    path = []

    for (u, v, i) in final_path:

        # Edge does not exist in org_graph, revert the coordinates
        # (unfortunately, there is a bug in osmnx with using undirected graph,
        # ~5% of one way streets have incorrect coords)
        if not org_graph.get_edge_data(u, v) or i not in org_graph.get_edge_data(u, v):
            inverse_edge = org_graph.get_edge_data(v, u)
            if inverse_edge and i in inverse_edge and 'geometry' in inverse_edge[i]:
                coords = list(inverse_edge[i]['geometry'].coords)[::-1]

                for (x, y) in coords:
                    path.append([y, x])

            continue

        # Edge does not have `geometry` field, return origin and end coordinates
        if 'geometry' not in org_graph.get_edge_data(u, v)[i]:
            x1 = org_graph.nodes[u]["x"]
            x2 = org_graph.nodes[v]["x"]

            y1 = org_graph.nodes[u]["y"]
            y2 = org_graph.nodes[v]["y"]

            path.extend([[y1, x1], [y2, x2]])
            continue

        # Edge does exists and has geometry and was hit in the right direction
        coords = list(org_graph.get_edge_data(u, v)[i]['geometry'].coords)

        for (x, y) in coords:
            path.append([y, x])

    return path
