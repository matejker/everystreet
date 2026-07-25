import { useEffect } from "react";
import PropTypes from "prop-types";
import { useLeaflet } from "react-leaflet";
import L from "leaflet";
import "leaflet.polyline.snakeanim/L.Polyline.SnakeAnim.js";

const SnakeAnim = ({ startAnimation, path }) => {
  const { map } = useLeaflet();

  useEffect(() => {
    if (!startAnimation) return;

    const polylineOptions = {
       color: 'red',
       weight: 3,
       opacity: 0.3,
       snakingSpeed: 150
     };

    const route = L.polyline(path, polylineOptions);
    map.fitBounds(route.getBounds());
    map.addLayer(route);
    route.snakeIn();
    route.on("snakestart snake snakeend", ev => {});

  }, [startAnimation]);

  return null;
};

SnakeAnim.propTypes = {
  startAnimation: PropTypes.bool.isRequired
};

export default SnakeAnim;
