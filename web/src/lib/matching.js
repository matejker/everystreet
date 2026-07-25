// Minimum-weight perfect matching over the odd-degree nodes.
//
// The old backend used networkx `max_weight_matching(G, maxcardinality=True)`
// on a complete graph built from *negated* distances. edmonds-blossom is a
// direct port of the same Van Rantwijk algorithm networkx uses, so we do the
// exact same trick: negate weights and ask for maximum cardinality.

import blossom from "edmonds-blossom";

// pairWeights: Array<[i, j, distance]> where i, j are dense integer indices.
// Returns Array<[i, j]> of matched index pairs.
export function minWeightMatching(pairWeights) {
  if (!pairWeights.length) return [];

  // Negate (min -> max) and round to integers for numerical stability, which
  // the blossom dual-variable termination relies on.
  const edges = pairWeights.map(([i, j, w]) => [i, j, -Math.round(w)]);

  const mate = blossom(edges, true); // maxCardinality = true -> perfect matching

  const pairs = [];
  for (let i = 0; i < mate.length; i += 1) {
    const j = mate[i];
    if (j > i) pairs.push([i, j]);
  }
  return pairs;
}
