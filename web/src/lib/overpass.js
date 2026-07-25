// Downloads the runnable street network for a polygon straight from the
// Overpass API. This replaces what osmnx did server-side in the old lambda.

// Same highway/access filter the old osmnx-based backend used, expressed as
// Overpass QL tag filters (osmnx built the very same query under the hood).
const WAY_FILTER =
  '["highway"]' +
  '["area"!~"yes"]' +
  '["highway"!~"bus_guideway|bus_stop|construction|cycleway|elevator|footway|' +
  "motorway|motorway_junction|motorway_link|escalator|proposed|construction|platform|raceway|rest_area|" +
  'path|service"]' +
  '["access"!~"customers|no|private"]' +
  '["public_transport"!~"platform"]' +
  '["fee"!~"yes"]' +
  '["foot"!~"no"]' +
  '["service"!~"drive-through|driveway|parking_aisle"]' +
  '["toll"!~"yes"]';

const ENDPOINTS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
];

// Builds the Overpass "poly" clause: a space separated "lat lon lat lon ..."
// string. Input ring is GeoJSON order ([lon, lat]).
function polyClause(lonLatRing) {
  return lonLatRing.map(([lon, lat]) => `${lat} ${lon}`).join(" ");
}

function buildQuery(lonLatRing) {
  const poly = polyClause(lonLatRing);
  return (
    "[out:json][timeout:180];" +
    `way${WAY_FILTER}(poly:"${poly}");` +
    "(._;>;);" +
    "out body;"
  );
}

async function post(endpoint, query) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `data=${encodeURIComponent(query)}`,
  });

  if (!res.ok) {
    throw new Error(`Overpass request failed (${res.status})`);
  }

  return res.json();
}

// Fetches raw OSM elements for the polygon, trying endpoints in order.
export async function fetchOsm(lonLatRing) {
  const query = buildQuery(lonLatRing);

  let lastError;
  for (const endpoint of ENDPOINTS) {
    try {
      const data = await post(endpoint, query);
      return data.elements || [];
    } catch (err) {
      lastError = err;
    }
  }

  throw lastError || new Error("Could not reach any Overpass endpoint");
}
