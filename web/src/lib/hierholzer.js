// Hierholzer's algorithm: finds an Eulerian circuit on a connected multigraph
// whose vertices all have even degree. Replaces the `mk-network` hierholzer
// used by the old Python backend.

// edges: Array<{ u, v }> (indices reference this array as edge ids)
// startNode: node id to start/end the circuit on
// Returns an ordered Array<{ node, edge }> describing the walk. The first entry
// has edge === -1; every subsequent entry's `edge` is the edge traversed to
// arrive at `node` from the previous entry's node.
export function hierholzer(edges, startNode) {
  const adjacency = new Map();
  const addHalf = (node, edgeId, to) => {
    if (!adjacency.has(node)) adjacency.set(node, []);
    adjacency.get(node).push({ edgeId, to });
  };

  edges.forEach((edge, id) => {
    addHalf(edge.u, id, edge.v);
    addHalf(edge.v, id, edge.u);
  });

  const used = new Array(edges.length).fill(false);
  const pointer = new Map();
  for (const node of adjacency.keys()) pointer.set(node, 0);

  const stack = [{ node: startNode, edge: -1 }];
  const circuit = [];

  while (stack.length) {
    const current = stack[stack.length - 1];
    const node = current.node;
    const neighbours = adjacency.get(node) || [];

    let ptr = pointer.get(node) || 0;
    while (ptr < neighbours.length && used[neighbours[ptr].edgeId]) ptr += 1;
    pointer.set(node, ptr);

    if (ptr < neighbours.length) {
      const { edgeId, to } = neighbours[ptr];
      used[edgeId] = true;
      stack.push({ node: to, edge: edgeId });
    } else {
      circuit.push(stack.pop());
    }
  }

  return circuit.reverse();
}
