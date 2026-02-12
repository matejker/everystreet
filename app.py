import os
import json
import traceback
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, Response

import networkx as nx
import osmnx as ox
from network import Network
from network.algorithms import hierholzer

from libs.tools import (
    get_odd_degree_nodes,
    get_shortest_distance_for_odd_degrees,
    min_matching,
    get_shortest_paths,
    map_osmnx_edges2integers,
    get_starting_node,
    convert_integer_path2osmnx_nodes,
    get_double_edge_heap,
    convert_path,
    convert_final_path_to_coordinates,
)
from libs.gpx_formatter import TEMPLATE, TRACE_POINT

app = Flask(__name__)

CUSTOM_FILTER = (
    '["highway"]["area"!~"yes"]["highway"!~"bridleway|bus_guideway|bus_stop|construction|cycleway|elevator|footway|'
    'motorway|motorway_junction|motorway_link|escalator|proposed|construction|platform|raceway|rest_area|'
    'path|service"]["access"!~"customers|no|private"]["public_transport"!~"platform"]'
    '["fee"!~"yes"]["foot"!~"no"]["service"!~"drive-through|driveway|parking_aisle"]["toll"!~"yes"]'
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate_route():
    """Generate an optimal every-street route for a given location."""
    data = request.get_json()
    location = data.get("location", "").strip()
    start_lat = data.get("start_lat")
    start_lon = data.get("start_lon")

    if not location:
        return jsonify({"error": "Location is required"}), 400

    try:
        # Fetch the street network from OSM
        org_graph = ox.graph_from_place(location, custom_filter=CUSTOM_FILTER)
        graph = ox.convert.to_undirected(org_graph)

        # Run the Chinese Postman algorithm
        odd_degree_nodes = get_odd_degree_nodes(graph)
        pair_weights = get_shortest_distance_for_odd_degrees(graph, odd_degree_nodes)
        matched_edges_with_weights = min_matching(pair_weights)

        single_edges = [(u, v) for u, v, k in graph.edges]
        added_edges = get_shortest_paths(graph, matched_edges_with_weights)
        edges = map_osmnx_edges2integers(graph, single_edges + added_edges)

        # Pick starting node
        if start_lat is not None and start_lon is not None:
            source = get_starting_node(graph, lat=float(start_lat), lon=float(start_lon))
        else:
            source = 0

        # Find Eulerian path
        network = Network(len(graph.nodes), edges, weighted=True)
        eulerian_path = hierholzer(network, source=source)
        converted_path = convert_integer_path2osmnx_nodes(eulerian_path, graph.nodes())
        double_edge_heap = get_double_edge_heap(org_graph)
        final_path = convert_path(graph, converted_path, double_edge_heap)

        # Convert to coordinates for Leaflet
        coordinates = convert_final_path_to_coordinates(org_graph, final_path)

        # Compute stats
        total_length = sum(
            org_graph.get_edge_data(u, v, {}).get(i, {}).get("length", 0)
            if org_graph.get_edge_data(u, v)
            else org_graph.get_edge_data(v, u, {}).get(i, {}).get("length", 0)
            for u, v, i in final_path
        )

        # Get center for map view
        nodes_data = org_graph.nodes(data=True)
        lats = [d["y"] for _, d in nodes_data]
        lons = [d["x"] for _, d in nodes_data]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        return jsonify({
            "coordinates": coordinates,
            "stats": {
                "total_length_km": round(total_length / 1000, 2),
                "num_edges": len(graph.edges),
                "num_nodes": len(graph.nodes),
                "num_edges_route": len(final_path),
            },
            "center": {"lat": center_lat, "lon": center_lon},
            "location": location,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/gpx", methods=["POST"])
def download_gpx():
    """Generate and return a GPX file for the computed route."""
    data = request.get_json()
    coordinates = data.get("coordinates", [])
    location = data.get("location", "EveryStreet Route")

    if not coordinates:
        return jsonify({"error": "No route coordinates provided"}), 400

    center_lat = sum(c[0] for c in coordinates) / len(coordinates)
    center_lon = sum(c[1] for c in coordinates) / len(coordinates)

    start_time = datetime(2024, 1, 1, 8, 0, 0)
    trace_points = "\n            ".join(
        TRACE_POINT.format(
            lat=coord[0],
            lon=coord[1],
            id=i,
            timestamp=(start_time + timedelta(seconds=i * 10)).isoformat() + "Z",
        )
        for i, coord in enumerate(coordinates)
    )

    gpx_content = TEMPLATE.format(
        name=location,
        center_lat=center_lat,
        center_lon=center_lon,
        trace_points=trace_points,
    )

    return Response(
        gpx_content,
        mimetype="application/gpx+xml",
        headers={"Content-Disposition": f"attachment; filename=everystreet_route.gpx"},
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
