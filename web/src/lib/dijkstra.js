// Dijkstra shortest paths on the undirected (multi) street graph, used to
// connect odd-degree nodes when solving the route inspection problem.

// Minimal binary min-heap keyed by numeric priority.
class MinHeap {
  constructor() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  push(node, priority) {
    const items = this.items;
    items.push({ node, priority });
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (items[parent].priority <= items[i].priority) break;
      [items[parent], items[i]] = [items[i], items[parent]];
      i = parent;
    }
  }

  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length) {
      items[0] = last;
      let i = 0;
      const n = items.length;
      while (true) {
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        let smallest = i;
        if (left < n && items[left].priority < items[smallest].priority) smallest = left;
        if (right < n && items[right].priority < items[smallest].priority) smallest = right;
        if (smallest === i) break;
        [items[smallest], items[i]] = [items[i], items[smallest]];
        i = smallest;
      }
    }
    return top;
  }
}

// adjacency: Map<nodeId, Array<{ to, length }>>
// Returns { dist: Map<nodeId, number>, prev: Map<nodeId, nodeId> }.
export function dijkstra(adjacency, source) {
  const dist = new Map();
  const prev = new Map();
  const heap = new MinHeap();

  dist.set(source, 0);
  heap.push(source, 0);

  while (heap.size) {
    const { node, priority } = heap.pop();
    if (priority > (dist.get(node) ?? Infinity)) continue;

    for (const { to, length } of adjacency.get(node) || []) {
      const candidate = priority + length;
      if (candidate < (dist.get(to) ?? Infinity)) {
        dist.set(to, candidate);
        prev.set(to, node);
        heap.push(to, candidate);
      }
    }
  }

  return { dist, prev };
}

// Reconstructs the node path from a `prev` map produced by dijkstra().
export function reconstructPath(prev, source, target) {
  const path = [target];
  let current = target;
  while (current !== source) {
    if (!prev.has(current)) return null; // unreachable
    current = prev.get(current);
    path.push(current);
  }
  return path.reverse();
}
