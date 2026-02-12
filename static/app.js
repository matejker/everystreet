// Initialize map centered on Edinburgh
const map = L.map("map").setView([55.94, -3.19], 13);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 19,
}).addTo(map);

let routeLayer = null;
let startMarker = null;
let lastRouteData = null;

const locationInput = document.getElementById("location");
const startLatInput = document.getElementById("start-lat");
const startLonInput = document.getElementById("start-lon");
const generateBtn = document.getElementById("generate-btn");
const clearStartBtn = document.getElementById("clear-start");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");

// Click on map to set starting point
map.on("click", function (e) {
    setStartingPoint(e.latlng.lat, e.latlng.lng);
});

function setStartingPoint(lat, lng) {
    startLatInput.value = lat.toFixed(6);
    startLonInput.value = lng.toFixed(6);
    clearStartBtn.style.display = "inline";

    if (startMarker) {
        startMarker.setLatLng([lat, lng]);
    } else {
        const icon = L.divIcon({
            className: "start-marker",
            iconSize: [14, 14],
            iconAnchor: [7, 7],
        });
        startMarker = L.marker([lat, lng], { icon: icon, draggable: true }).addTo(map);
        startMarker.bindTooltip("Starting point", { direction: "top", offset: [0, -10] });
        startMarker.on("dragend", function (e) {
            const pos = e.target.getLatLng();
            startLatInput.value = pos.lat.toFixed(6);
            startLonInput.value = pos.lng.toFixed(6);
        });
    }
}

clearStartBtn.addEventListener("click", function () {
    startLatInput.value = "";
    startLonInput.value = "";
    clearStartBtn.style.display = "none";
    if (startMarker) {
        map.removeLayer(startMarker);
        startMarker = null;
    }
});

function showStatus(message, type) {
    statusEl.style.display = "block";
    statusEl.className = "status " + type;
    if (type === "loading") {
        statusEl.innerHTML = '<span class="spinner"></span>' + message;
    } else {
        statusEl.textContent = message;
    }
}

function hideStatus() {
    statusEl.style.display = "none";
}

// Generate route
generateBtn.addEventListener("click", async function () {
    const location = locationInput.value.trim();
    if (!location) {
        showStatus("Please enter a location.", "error");
        return;
    }

    generateBtn.disabled = true;
    statsEl.style.display = "none";
    showStatus("Fetching street network and computing optimal route...", "loading");

    if (routeLayer) {
        map.removeLayer(routeLayer);
        routeLayer = null;
    }

    const payload = { location: location };
    if (startLatInput.value && startLonInput.value) {
        payload.start_lat = parseFloat(startLatInput.value);
        payload.start_lon = parseFloat(startLonInput.value);
    }

    try {
        const response = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Failed to generate route");
        }

        lastRouteData = result;
        displayRoute(result);
        showStatus("Route generated successfully!", "success");
        setTimeout(hideStatus, 3000);
    } catch (err) {
        showStatus("Error: " + err.message, "error");
    } finally {
        generateBtn.disabled = false;
    }
});

function displayRoute(data) {
    const coords = data.coordinates;

    // Draw the route
    routeLayer = L.polyline(coords, {
        color: "#e74c3c",
        weight: 3,
        opacity: 0.7,
    }).addTo(map);

    // Fit map to route bounds
    map.fitBounds(routeLayer.getBounds(), { padding: [30, 30] });

    // Update stats
    document.getElementById("stat-distance").textContent = data.stats.total_length_km + " km";
    document.getElementById("stat-edges").textContent = data.stats.num_edges;
    document.getElementById("stat-nodes").textContent = data.stats.num_nodes;
    document.getElementById("stat-route-edges").textContent = data.stats.num_edges_route;
    statsEl.style.display = "block";
}

// Enter key triggers generate
locationInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
        generateBtn.click();
    }
});

// Download GPX
document.getElementById("download-gpx").addEventListener("click", async function () {
    if (!lastRouteData) return;

    try {
        const response = await fetch("/api/gpx", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                coordinates: lastRouteData.coordinates,
                location: lastRouteData.location,
            }),
        });

        if (!response.ok) throw new Error("Failed to generate GPX");

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "everystreet_route.gpx";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (err) {
        showStatus("Error downloading GPX: " + err.message, "error");
    }
});
