// Web Worker entry point. Runs the (CPU heavy) route generation off the main
// thread so the map/UI stay responsive while a route is being computed.

/* eslint-disable no-restricted-globals */
import { generateRoute } from "./routeInspection";

self.onmessage = async (event) => {
  const { lonLatRing, name } = event.data;

  try {
    const payload = await generateRoute(lonLatRing, name, (progress) => {
      self.postMessage({ type: "progress", progress });
    });
    self.postMessage({ type: "result", payload });
  } catch (err) {
    self.postMessage({ type: "error", message: err.message || String(err) });
  }
};
