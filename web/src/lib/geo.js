// Geometry helpers used by the in-browser route engine.

const EARTH_RADIUS = 6371009; // metres, same value osmnx uses

const toRad = (deg) => (deg * Math.PI) / 180;

// Great-circle distance between two [lat, lon] points, in metres.
export function haversine(a, b) {
  const [lat1, lon1] = a;
  const [lat2, lon2] = b;

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const sinLat = Math.sin(dLat / 2);
  const sinLon = Math.sin(dLon / 2);

  const h =
    sinLat * sinLat +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * sinLon * sinLon;

  return 2 * EARTH_RADIUS * Math.asin(Math.min(1, Math.sqrt(h)));
}

// Total length in metres of a poly-line given as a list of [lat, lon] points.
export function lineLength(coords) {
  let total = 0;
  for (let i = 1; i < coords.length; i += 1) {
    total += haversine(coords[i - 1], coords[i]);
  }
  return total;
}

// Approximate planar area (km^2) of a polygon given as GeoJSON [lon, lat] ring.
// Mirrors the guard the old Python lambda used (`area_of_polygon`).
export function polygonAreaKm2(lonLatRing) {
  const latDist = (Math.PI * EARTH_RADIUS) / 180.0;

  const x = lonLatRing.map(([lon, lat]) => lon * latDist * Math.cos(toRad(lat)));
  const y = lonLatRing.map(([, lat]) => lat * latDist);

  let area = 0.0;
  const n = x.length;
  for (let i = 0; i < n; i += 1) {
    const prev = (i - 1 + n) % n;
    const next = (i + 1) % n;
    area += x[i] * (y[next] - y[prev]);
  }

  return Math.abs(area) / 2.0 / 1e6;
}

// Ray-casting point-in-polygon test.
//   latLon: [lat, lon] point
//   lonLatRing: polygon ring in GeoJSON order ([lon, lat] points)
export function pointInPolygon(latLon, lonLatRing) {
  const x = latLon[1]; // lon
  const y = latLon[0]; // lat

  let inside = false;
  for (let i = 0, j = lonLatRing.length - 1; i < lonLatRing.length; j = i, i += 1) {
    const xi = lonLatRing[i][0];
    const yi = lonLatRing[i][1];
    const xj = lonLatRing[j][0];
    const yj = lonLatRing[j][1];

    const intersects =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }

  return inside;
}

// Centroid ([lat, lon]) of a list of [lat, lon] points.
export function centroid(latLonPoints) {
  if (!latLonPoints.length) return [0, 0];
  let lat = 0;
  let lon = 0;
  for (const [la, lo] of latLonPoints) {
    lat += la;
    lon += lo;
  }
  return [lat / latLonPoints.length, lon / latLonPoints.length];
}
