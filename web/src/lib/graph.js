// Turns raw Overpass elements into a routable undirected multigraph, roughly
// mirroring what osmnx did: split ways at intersections, keep the geometry of
// each segment, compute segment lengths and keep the largest connected
// component so the route-inspection solver always gets a connected graph.

import { haversine, pointInPolygon } from "./geo";

// Builds { nodes, edges } from Overpass elements.
//   nodes: Map<nodeId, [lat, lon]>
//   edges: Array<{ u, v, length, geometry: [[lat, lon], ...] }>  (u -> v order)
// If `boundaryRing` (GeoJSON [lon, lat] polygon ring) is given, the graph is
// clipped to it the same way osmnx did (truncate_by_edge=False): nodes outside
// the polygon are dropped, so only edges with both endpoints inside survive.
export function buildGraph(elements, boundaryRing) {
  const coords = new Map(); // nodeId -> [lat, lon]
  const ways = [];

  for (const el of elements) {
    if (el.type === "node") {
      coords.set(el.id, [el.lat, el.lon]);
    } else if (el.type === "way" && Array.isArray(el.nodes)) {
      ways.push(el);
    }
  }

  // Count how often each node is referenced so we can find intersections.
  const useCount = new Map();
  for (const way of ways) {
    for (const nodeId of way.nodes) {
      useCount.set(nodeId, (useCount.get(nodeId) || 0) + 1);
    }
  }

  const isVertex = (nodeId, way, index) => {
    if (index === 0 || index === way.nodes.length - 1) return true; // way endpoint
    return (useCount.get(nodeId) || 0) > 1; // shared junction
  };

  const nodes = new Map();
  const edges = [];

  const addVertex = (nodeId) => {
    if (!nodes.has(nodeId) && coords.has(nodeId)) {
      nodes.set(nodeId, coords.get(nodeId));
    }
  };

  for (const way of ways) {
    const refs = way.nodes.filter((id) => coords.has(id));
    if (refs.length < 2) continue;

    let startId = refs[0];
    let geometry = [coords.get(startId)];

    for (let i = 1; i < refs.length; i += 1) {
      const nodeId = refs[i];
      const point = coords.get(nodeId);
      geometry.push(point);

      if (isVertex(nodeId, { nodes: refs }, i)) {
        if (nodeId !== startId && geometry.length >= 2) {
          let length = 0;
          for (let j = 1; j < geometry.length; j += 1) {
            length += haversine(geometry[j - 1], geometry[j]);
          }

          if (length > 0) {
            addVertex(startId);
            addVertex(nodeId);
            edges.push({ u: startId, v: nodeId, length, geometry });
          }
        }

        startId = nodeId;
        geometry = [point];
      }
    }
  }

  const clipped = boundaryRing
    ? clipToPolygon(nodes, edges, boundaryRing)
    : { nodes, edges };

  return largestComponent(clipped.nodes, clipped.edges);
}

// Keeps only edges whose both endpoints lie inside the polygon (mirrors
// osmnx `truncate_by_edge=False`), then drops any now-orphaned nodes.
function clipToPolygon(nodes, edges, boundaryRing) {
  const inside = new Map();
  for (const [id, point] of nodes) {
    inside.set(id, pointInPolygon(point, boundaryRing));
  }

  const keptEdges = edges.filter((e) => inside.get(e.u) && inside.get(e.v));

  const keptNodes = new Map();
  for (const e of keptEdges) {
    keptNodes.set(e.u, nodes.get(e.u));
    keptNodes.set(e.v, nodes.get(e.v));
  }

  return { nodes: keptNodes, edges: keptEdges };
}

// Keeps only the largest connected component (route inspection needs a
// connected graph, just like networkx/osmnx required).
function largestComponent(nodes, edges) {
  const adj = new Map();
  for (const id of nodes.keys()) adj.set(id, []);
  edges.forEach((edge, index) => {
    adj.get(edge.u).push({ to: edge.v, index });
    adj.get(edge.v).push({ to: edge.u, index });
  });

  const componentOf = new Map();
  let bestComponent = -1;
  let bestSize = 0;
  let componentId = 0;

  for (const start of nodes.keys()) {
    if (componentOf.has(start)) continue;

    const stack = [start];
    const members = [];
    componentOf.set(start, componentId);

    while (stack.length) {
      const node = stack.pop();
      members.push(node);
      for (const { to } of adj.get(node)) {
        if (!componentOf.has(to)) {
          componentOf.set(to, componentId);
          stack.push(to);
        }
      }
    }

    if (members.length > bestSize) {
      bestSize = members.length;
      bestComponent = componentId;
    }
    componentId += 1;
  }

  const keptNodes = new Map();
  for (const [id, point] of nodes) {
    if (componentOf.get(id) === bestComponent) keptNodes.set(id, point);
  }

  const keptEdges = edges.filter(
    (edge) => componentOf.get(edge.u) === bestComponent
  );

  return { nodes: keptNodes, edges: keptEdges };
}
