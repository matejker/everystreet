import networkx as nx
import numpy as np
from itertools import combinations
from network import Network
from heapq import heappop, heapify


def get_odd_degree_nodes(graph):
    # Note: why all osmnx nodes has even degree even they are odd and we need to divide by 2 ¯\_(ツ)_/¯
    return [v for v, d in graph.degree() if d % 2 == 1] # [v for v, d in graph.degree() if (d / 2) % 2 == 1] +


def map_osmnx_nodes2integers(graph):
    nodes_list = list(graph.nodes)
    nodes_list.sort()

    return len(nodes_list), nodes_list


def map_osmnx_edges2integers(graph, edges):
    n, nodes_list = map_osmnx_nodes2integers(graph)

    converted_edges = []
    for v, u, _ in edges:
        # TODO: is .index() the best option, what about using dict()
        converted_edges.append((nodes_list.index(v), nodes_list.index(u)))

    return converted_edges


def create_weighted_complete_graph(pair_weights):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(e[0], e[1], - np.round(w, 2)) for e, w in pair_weights.items()])

    return graph


def max_matching(pair_weights):
    complete_graph = create_weighted_complete_graph(pair_weights)
    matching = nx.algorithms.max_weight_matching(complete_graph, True)

    matching_weights = []
    for v, u in matching:
        if (v, u) in pair_weights:
            matching_weights.append((v, u, np.round(pair_weights[v, u], 2)))
        else:
            matching_weights.append((v, u, np.round(pair_weights[u, v], 2)))

    return matching_weights


def get_single_edges(graph):
    edges_list = []
    for (v, u, d) in graph.edges(data=True):
        e = (v, u, np.round(d['length'], 2))
        r_e = (e[1], e[0], e[2])

        if e not in edges_list and r_e not in edges_list:
            edges_list.append(e)

    return edges_list


def get_shortest_distance_for_odd_degrees(graph, odd_degree_nodes):
    pairs = combinations(odd_degree_nodes, 2)

    return {(v, u): nx.dijkstra_path_length(graph, v, u, weight='length') for v, u in pairs}


def networkx2network(graph):
    """Converts networkx.Graph() into Network

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

    # weighted_edges = [(v, u, np.round(d['weight'], 2)) for (v, u, d) in graph.edges(data=True) if 'weight' in d]  # noqa
    # weighted = len(weighted_edges) > 0
    #
    # if weighted:
    #     unweighted_edges = [(nodes.index(v), nodes.index(u), 1) for (v, u, d) in graph.edges(data=True) if 'weight' not in d]  # noqa
    #     edges = weighted_edges + unweighted_edges
    # else:
    #     edges = [(nodes.index(v), nodes.index(u)) for v, u in graph.edges()]
    #
    # print(n, edges)

    return Network(n, edges, directed=False, weighted=True)


def get_double_edge_heap(graph):
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
        if current_node == perv_node and not graph.get_edge_data(current_node, perv_node):
            current_node = path.pop()

        edge = (perv_node, current_node)

        # Edge doesn't exist, getting shortest distance extending the path
        if not graph.get_edge_data(*edge):
            shortest_path = nx.shortest_path(graph, perv_node, current_node, weight='length')[::-1]
            path.extend(shortest_path[:-1])

            current_node = shortest_path[-2]
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
