// Orchestrates the whole every-street route generation in the browser.
// Mirrors the old Python `route_generator()` + `tools.py` + `stats.py`.

import { fetchOsm } from "./overpass";
import { buildGraph } from "./graph";
import { dijkstra, reconstructPath } from "./dijkstra";
import { minWeightMatching } from "./matching";
import { hierholzer } from "./hierholzer";
import { lineLength, centroid, polygonAreaKm2 } from "./geo";

export const MAX_POLYGON_AREA = 3; // km^2, same guard as the old backend

const noop = () => {};

// Undirected key for an edge between two nodes.
const pairKey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);

function degrees(nodes, edges) {
  const deg = new Map();
  for (const id of nodes.keys()) deg.set(id, 0);
  for (const { u, v } of edges) {
    deg.set(u, (deg.get(u) || 0) + 1);
    deg.set(v, (deg.get(v) || 0) + 1);
  }
  return deg;
}

// Builds Map<nodeId, Array<{ to, length }>> for Dijkstra (keeps multi-edges).
function buildAdjacency(nodes, edges) {
  const adjacency = new Map();
  for (const id of nodes.keys()) adjacency.set(id, []);
  for (const { u, v, length } of edges) {
    adjacency.get(u).push({ to: v, length });
    adjacency.get(v).push({ to: u, length });
  }
  return adjacency;
}

// Groups base edges by node pair so we can pick the shortest one when a street
// needs to be traversed twice (equivalent to the old double-edge heap).
function indexEdgesByPair(edges) {
  const byPair = new Map();
  edges.forEach((edge, index) => {
    const key = pairKey(edge.u, edge.v);
    if (!byPair.has(key)) byPair.set(key, []);
    byPair.get(key).push(index);
  });
  return byPair;
}

function walkToCoordinates(circuit, allEdges) {
  const path = [];

  for (let i = 1; i < circuit.length; i += 1) {
    const fromNode = circuit[i - 1].node;
    const edge = allEdges[circuit[i].edge];
    if (!edge) continue;

    let geometry = edge.geometry;
    if (edge.u !== fromNode) geometry = geometry.slice().reverse();

    const startAt = path.length === 0 ? 0 : 1; // avoid duplicating the join point
    for (let j = startAt; j < geometry.length; j += 1) {
      path.push(geometry[j]);
    }
  }

  return path;
}

// Solves the route inspection problem on an already built graph.
export function solve({ nodes, edges }, onProgress = noop) {
  if (edges.length === 0) {
    throw new Error("No runnable streets were found in this area.");
  }

  const deg = degrees(nodes, edges);
  const oddNodes = [...deg.entries()]
    .filter(([, d]) => d % 2 === 1)
    .map(([id]) => id);

  onProgress({ stage: "matching", oddNodes: oddNodes.length });

  const adjacency = buildAdjacency(nodes, edges);
  const byPair = indexEdgesByPair(edges);

  // Extra edge instances added to make every vertex even degree.
  const addedEdges = [];

  if (oddNodes.length > 0) {
    // Shortest paths from every odd node (dist for pair weights, prev for
    // reconstructing the concrete street path we duplicate).
    const prevByNode = new Map();
    const distByNode = new Map();
    oddNodes.forEach((node, i) => {
      const { dist, prev } = dijkstra(adjacency, node);
      prevByNode.set(node, prev);
      distByNode.set(node, dist);
      if (i % 10 === 0) {
        onProgress({ stage: "shortest-paths", done: i, total: oddNodes.length });
      }
    });

    // Complete graph over odd nodes with shortest-distance weights.
    const pairWeights = [];
    for (let i = 0; i < oddNodes.length; i += 1) {
      const dist = distByNode.get(oddNodes[i]);
      for (let j = i + 1; j < oddNodes.length; j += 1) {
        const w = dist.get(oddNodes[j]);
        if (w != null && Number.isFinite(w)) pairWeights.push([i, j, w]);
      }
    }

    const matched = minWeightMatching(pairWeights);

    for (const [i, j] of matched) {
      const source = oddNodes[i];
      const target = oddNodes[j];
      const nodePath = reconstructPath(prevByNode.get(source), source, target);
      if (!nodePath) continue;

      for (let k = 1; k < nodePath.length; k += 1) {
        const a = nodePath[k - 1];
        const b = nodePath[k];
        const candidates = byPair.get(pairKey(a, b)) || [];
        if (!candidates.length) continue;

        // Duplicate the shortest street edge between a and b.
        let best = candidates[0];
        for (const idx of candidates) {
          if (edges[idx].length < edges[best].length) best = idx;
        }
        addedEdges.push(edges[best]);
      }
    }
  }

  const allEdges = edges.concat(addedEdges);

  onProgress({ stage: "eulerian" });
  const circuit = hierholzer(allEdges, allEdges[0].u);
  const path = walkToCoordinates(circuit, allEdges);

  const streetLengthTotal = edges.reduce((s, e) => s + e.length, 0);
  const pathLengthTotal = allEdges.reduce((s, e) => s + e.length, 0);

  return {
    path,
    statistics: {
      path_stats: {
        street_length_total: Math.round((streetLengthTotal / 1000) * 100) / 100,
        path_length_total: Math.round((pathLengthTotal / 1000) * 100) / 100,
      },
      network_stats: {
        center: centroid(path),
        diameter: 0,
        radius: 0,
      },
    },
    status: "public",
  };
}

// Full pipeline: validate -> download OSM -> build graph -> solve.
// `lonLatRing` is a GeoJSON polygon ring ([lon, lat] points).
export async function generateRoute(lonLatRing, name, onProgress = noop) {
  const area = polygonAreaKm2(lonLatRing);
  if (area > MAX_POLYGON_AREA) {
    throw new Error(
      `Selected area is ${area.toFixed(2)} km², which is bigger than the ${MAX_POLYGON_AREA} km² limit.`
    );
  }

  onProgress({ stage: "download" });
  const elements = await fetchOsm(lonLatRing);

  onProgress({ stage: "build-graph" });
  const graph = buildGraph(elements, lonLatRing);

  const payload = solve(graph, onProgress);
  payload.statistics.name = name;
  return payload;
}
