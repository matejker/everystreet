import React, { useEffect, useState } from "react";

// High-level steps shown as a little "agent is working" checklist.
const STEPS = [
  { key: "download", label: "Fetching streets from OpenStreetMap" },
  { key: "build-graph", label: "Building the street network" },
  { key: "solve", label: "Optimising your route" },
  { key: "eulerian", label: "Assembling the final route" },
];

// Raw engine stages -> checklist step index.
const STAGE_TO_STEP = {
  download: 0,
  "build-graph": 1,
  matching: 2,
  "shortest-paths": 2,
  eulerian: 3,
};

// Rotating flavour text per step — the smoke & mirrors.
const FLAVOR = {
  download: [
    "Negotiating with the Overpass API",
    "Downloading every street, alley and cul-de-sac",
    "Waking up the map servers",
    "Reading the neighbourhood, one road at a time",
  ],
  "build-graph": [
    "Splitting ways at every intersection",
    "Measuring each segment down to the metre",
    "Trimming streets outside your area",
    "Finding the largest connected neighbourhood",
  ],
  solve: [
    "Hunting down odd intersections",
    "Computing thousands of shortest paths",
    "Running blossom matching on the dead-ends",
    "Solving the Chinese Postman Problem",
    "Squeezing out every redundant metre",
    "Consulting Edmonds & Johnson (1973)",
  ],
  eulerian: [
    "Stitching together the Eulerian circuit",
    "Making sure you won't miss a single street",
    "Adding the finishing touches",
  ],
};

const RouteLoader = ({ progress }) => {
  const stage = progress && progress.stage ? progress.stage : "download";
  const currentStep = STAGE_TO_STEP[stage] ?? 0;
  const stepKey = STEPS[currentStep].key;

  const [tick, setTick] = useState(0);

  // Reset the rotation whenever we move to a new step so it feels responsive.
  useEffect(() => {
    setTick(0);
  }, [stepKey]);

  // Cycle the flavour text.
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1900);
    return () => clearInterval(id);
  }, []);

  const pool = FLAVOR[stepKey] || FLAVOR.solve;

  // Prefer the real numeric signal when we have it.
  let flavor;
  if (stage === "shortest-paths" && progress && progress.total) {
    flavor = `Computing shortest paths — ${progress.done} / ${progress.total}`;
  } else {
    flavor = pool[tick % pool.length];
  }

  return (
    <div className="route-loader" role="status" aria-live="polite">
      <div className="route-loader__headline">
        <span className="route-loader__spin" aria-hidden="true" />
        <span className="route-loader__flavor" key={`${stepKey}-${flavor}`}>
          {flavor}
        </span>
        <span className="route-loader__ellipsis" aria-hidden="true">
          <i>.</i>
          <i>.</i>
          <i>.</i>
        </span>
      </div>
    </div>
  );
};

export default RouteLoader;
