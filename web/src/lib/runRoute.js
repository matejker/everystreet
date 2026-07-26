// Thin client-side wrapper around the route worker. Falls back to running the
// engine on the main thread if Web Workers aren't available.

import { generateRoute } from "./routeInspection";

export function runRoute(
  lonLatRing,
  name,
  onProgress = () => {},
  startLatLon = null
) {
  if (typeof Worker === "undefined") {
    return generateRoute(lonLatRing, name, onProgress, startLatLon);
  }

  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./routeWorker.js", import.meta.url));

    worker.onmessage = (event) => {
      const { type } = event.data;
      if (type === "progress") {
        onProgress(event.data.progress);
      } else if (type === "result") {
        resolve(event.data.payload);
        worker.terminate();
      } else if (type === "error") {
        reject(new Error(event.data.message));
        worker.terminate();
      }
    };

    worker.onerror = (err) => {
      reject(err);
      worker.terminate();
    };

    worker.postMessage({ lonLatRing, name, startLatLon });
  });
}
