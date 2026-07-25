import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import RouteMap from "../components/Map";
import { SERVICE_URL } from "../config"
import axios from 'axios';
import polyline from 'polyline';

export const ShowRoute = () => {
    let { routeId } = useParams();

    const [pending, setPending] = useState(true);
    const [data, setData] = useState({});
    const [path, setPath] = useState([[0, 0], [1, 1]]);
    const [stats, setStats] = useState({
        path_length_total: 0,
        street_length_total: 0,
        center: [0, 0],
        diameter: 0,
        radius: 0,
        name: ''
    });

    useEffect(async () => {
        async function getData() {
            const result = await axios.get(
                `${SERVICE_URL}/route/${routeId}`,
                { headers: {'Content-Type': 'application/json'} }
            );

            if (result.data) {
                setPending(false);
                setData(result.data);
                if (Array.isArray(result.data.path)){
                    setPath(result.data.path);
                } else {
                    setPath(polyline.decode(result.data.path));
                }
                setStats({
                    path_length_total: result.data.statistics.path_stats.path_length_total,
                    street_length_total: result.data.statistics.path_stats.street_length_total,
                    center: result.data.statistics.network_stats.center,
                    diameter: result.data.statistics.network_stats.diameter,
                    radius: result.data.statistics.network_stats.radius,
                    name: result.data.statistics.name || ''
                });
            }
        }
        getData();

    }, [routeId]);

    return (
        <>
            <h2> { stats.name.split(', ', 2).join(', ') || `Route (${stats.center.join(', ')})` }</h2>
            { !data && <p>Wait I m Loading</p> }
            { pending && <p>Route is being generated</p>}
            { !pending && <>
                <RouteMap position={stats.center} path={path} />
                <ul>
                    <li>Total street length: { stats.street_length_total }km</li>
                    <li>Route length: { stats.path_length_total }km</li>
                    <li>Efficiency: +{ Math.round(1000 * (stats.path_length_total / stats.street_length_total  - 1)) / 10 }%</li>
                    <li>Diameter: { stats.diameter }km</li>
                    <li>Radius: { stats.radius }km</li>
                    <li><a href={`${SERVICE_URL}/route/gpx/${routeId}`} download={`everystreet-${routeId}.gpx`}>GPX file</a></li>
                </ul>
            </>}
        </>
    );
}
