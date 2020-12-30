from osmnx.plot import plot_graph, _save_and_show


def plot_graph_route(
    graph,
    route,
    route_color="r",
    route_linewidth=4,
    route_alpha=0.5,
    orig_dest_size=100,
    ax=None,
    **pg_kwargs
):
    """Plot a route along a graph. Allowing routes for multigraphs (two edges for the same nodes)
    Args:
    graph (networkx.MultiDiGraph) input graph
    route (list): route as a list of triples (node1, node2, key of the edge)
    route_color (string) color of the route
    route_linewidth (int): width of the route line
    route_alpha (float): opacity of the route line
    orig_dest_size (int): size of the origin and destination nodes
    ax  (matplotlib axis): if not None, plot route on this preexisting axis instead of creating a new fig, ax
      and drawing the underlying graph
    pg_kwargs: keyword arguments to pass to plot_graph

    Returns
    fig, ax (tuple): matplotlib figure, axis
    """
    if ax is None:
        # plot the graph but not the route, and override any user show/close
        # args for now: we'll do that later
        override = {"show", "save", "close"}
        kwargs = {k: v for k, v in pg_kwargs.items() if k not in override}
        fig, ax = plot_graph(graph, show=False, save=False, close=False, **kwargs)
    else:
        fig = ax.figure

    # scatterplot origin and destination points (first/last nodes in route)
    x = (graph.nodes[route[0][0]]["x"], graph.nodes[route[-1][1]]["x"])
    y = (graph.nodes[route[0][0]]["y"], graph.nodes[route[-1][1]]["y"])
    ax.scatter(x, y, s=orig_dest_size, c=route_color, alpha=route_alpha, edgecolor="none")

    # assemble the route edge geometries' x and y coords then plot the line
    x = []
    y = []
    for r in route:
        u, v, d = r
        data = graph.get_edge_data(u, v) or graph.get_edge_data(v, u)
        if d not in data:
            d = 0

        data = data[d]
        if "geometry" in data:
            # if geometry attribute exists, add all its coords to list
            xs, ys = data["geometry"].xy
            x.extend(xs)
            y.extend(ys)
        else:
            # otherwise, the edge is a straight line from node to node
            x.extend((graph.nodes[u]["x"], graph.nodes[v]["x"]))
            y.extend((graph.nodes[u]["y"], graph.nodes[v]["y"]))
    ax.plot(x, y, c=route_color, lw=route_linewidth, alpha=route_alpha)

    # save and show the figure as specified, passing relevant kwargs
    sas_kwargs = {"save", "show", "close", "filepath", "file_format", "dpi"}
    kwargs = {k: v for k, v in pg_kwargs.items() if k in sas_kwargs}
    fig, ax = _save_and_show(fig, ax, **kwargs)
    return fig, ax
