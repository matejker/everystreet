// Builds a GPX track from a route path and triggers a browser download.
// `path` is an array of [lat, lon] points (as produced by the route engine).

const escapeXml = (value) =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");

// Returns a GPX 1.1 document string with a single track segment.
export function buildGpx(path, name = "everystreet route") {
  const trackName = escapeXml(name || "everystreet route");
  const points = (path || [])
    .filter((p) => Array.isArray(p) && p.length >= 2)
    .map(([lat, lon]) => `      <trkpt lat="${lat}" lon="${lon}"></trkpt>`)
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="everystreet" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata>
    <name>${trackName}</name>
  </metadata>
  <trk>
    <name>${trackName}</name>
    <trkseg>
${points}
    </trkseg>
  </trk>
</gpx>
`;
}

// Turns a route name into a safe-ish file slug.
const slugify = (name) =>
  (name || "route")
    .split(",")[0]
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "route";

// Generates the GPX and prompts the browser to download it.
export function downloadGpx(path, name) {
  const gpx = buildGpx(path, name);
  const blob = new Blob([gpx], { type: "application/gpx+xml" });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `everystreet-${slugify(name)}.gpx`;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  URL.revokeObjectURL(url);
}
