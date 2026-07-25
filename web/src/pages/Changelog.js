import React from "react";

export const Changelog = () => {
  return (
      <>
        <h3>Changelog</h3>
        <pre>{`
v3.0.0 Browser-only rewrite (2026)
  - Route generation now runs entirely in the browser; the Python backend is gone
  - Street data is fetched directly from the Overpass API
  - The route inspection solver (Dijkstra, blossom matching, Hierholzer)
    was ported to JavaScript and runs in a Web Worker
  - Refreshed UI and copy
---------
v2.0.0 Google Polyline Algorithm & AWS update (April 2021)
  - Save every street's path as an Encoded Polyline Algorithm Format
    https://developers.google.com/maps/documentation/utilities/polylinealgorithm
  - Refactoring and dockerizing backend
---------
v1.1.0 Adjust OSM highway selection
 - Based on troyml42 idea, we modified the OSM query, more details on GitHub issue:
 https://github.com/matejker/everystreet/issues/3#issuecomment-739417939
---------
v1.0.0 First public release (August 2020)
 - Route generation on neighbourhood given by polygon or Nominatim location
 - Backend powered with OSMnx network on drive layer
 - React web app
        `}</pre>

      </>
  );
}
