import React, { useState } from "react";
import { Map, TileLayer, Polyline } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import SnakeAnim from "./SnakeAnim";

const RouteMap = ({ position, path }) => {
  const [startAnimation, setStartAnimation] = useState(false);
  const startSnake = () => setStartAnimation(!startAnimation);

  return (
    <>
      <Map style={{ height: "50vh" }} animate={true} zoom={15} bounds={path}>
        <TileLayer
          attribution='&amp;copy <a href="https://osm.org/copyright">OpenStreetMap</a>'
          url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
        />
        <SnakeAnim startAnimation={startAnimation} path={path} />
        <Polyline color="grey" positions={path} opacity={0.4} />
      </Map>
      <p></p>
      <button onClick={startSnake}>Run the route!</button>
    </>
  );
};

export default RouteMap;
