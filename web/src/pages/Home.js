import React, { useState } from "react";
import { Map, TileLayer, FeatureGroup, Polygon } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";
import "leaflet-draw/dist/leaflet.draw.css";
import { OPEN_STREET_MAP } from "../config";
import { Spacer } from '../components/constants';
import axios from 'axios';
import RouteMap from "../components/Map";
import RouteLoader from "../components/RouteLoader";
import { runRoute } from "../lib/runRoute";



//const postRequest = (polygon, name, history) => {
//
//    const message = getMessagePayload(polygon, name);
//    const insert = async () => {
//        let routeId = 'no-id';
//        await axios.post(
//            `${SERVICE_URL}/route`,
//            JSON.stringify(message),
//            { headers: {'Content-Type': 'application/json'}}
//        ).then(
//            (response) => {
//                if (!('error' in response.data) && 'route_id' in response.data) {
//                    routeId = response.data.route_id;
//                    history.push(`/route/${routeId}`)
//                }
//            },
//        ).catch(error => {
//            const defaultMessage = 'Route could not be generated, try again!'
//            const mess = error.response.data.user_error + '\n' + defaultMessage || defaultMessage;
//            alert(mess);
//            window.location.replace("/");  // Refreshing and redirecting to /
//        });
//    };
//
//    insert();
//
//};

export const Home = () => {
    // A / B testing
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const page_type = urlParams.get('page_type');

    const defaultNeighbourhood = (page_type === 'B') ? 'Larchmont, New York' : 'The Grange, Edinburgh';
    const defaultCenter = (page_type === 'B') ? ["40.9278769", "-73.7517983"] :["55.9325", "-3.1847"];

    const [polygon, setPolygon] = useState([]);
    const [name, setName] = useState();
    const [polygonReversed, setPolygonReversed] = useState([]);
    const [buttonDisabled, setButtonDisabled] = useState(true);
    const [loading, setLoading] = useState(false);
    const [noData, setNoData] = useState(false);
    const [progress, setProgress] = useState(null);

    const [neighbourhood, setNeighbourhood] = useState(defaultNeighbourhood);
    const [centerPoint, setCenterPoint] = useState(defaultCenter);
//    const [isBackendWorking, setIsBackendWorking] = useState(undefined);

    const [requestSent, setRequestSent] = useState(false);
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

//    useEffect(() => {
//        async function fetchData() {
//        // You can await here
//        const response = await axios.get(
//                `${SERVICE_URL}/health`,
//                { headers: {'Content-Type': 'application/json'} }
//            ).then(
//                (response) => {
//                if ('status' in response.data) {
//                    setIsBackendWorking(response.data.status === "ok");
//                }
//            }
//            ).catch(error => {
//                setIsBackendWorking(false);
//            });
//        }
//        fetchData();
//    });

    const postRequest = (polygon, name) => {
        const run = async () => {
            try {
                const result = await runRoute(polygon, name, (p) => setProgress(p));

                setPending(false);
                setData(result);
                setPath(result.path);
                setStats({
                    path_length_total: result.statistics.path_stats.path_length_total,
                    street_length_total: result.statistics.path_stats.street_length_total,
                    center: result.statistics.network_stats.center,
                    diameter: result.statistics.network_stats.diameter,
                    radius: result.statistics.network_stats.radius,
                    name: result.statistics.name || ''
                });
                setRequestSent(true);
            } catch (err) {
                alert(err.message || 'Route could not be generated, try again!');
                setButtonDisabled(false);
            } finally {
                setLoading(false);
                setProgress(null);
            }
        };
        run();
    };

    const reverseList = (list) => {
        const newlist = [];
        for (const i in list)
           newlist[i] = [list[i][1], list[i][0]];

        return newlist;
    }

    const getNeighbourhood = (e) => {
        e.preventDefault();
        const getData = async () => {
            const result = await axios.get(
                `${OPEN_STREET_MAP}?q=${neighbourhood}&polygon_geojson=1&format=json&limit=1`,
                { headers: {'Content-Type': 'application/json'} }
                );

            const data = result.data[0]

            if (!data) {
                setNoData(true);
                return false
            }

            const geojson = data.geojson;

            setName(data.display_name)

            if (geojson.type === 'Polygon' || geojson.type === 'MultiPolygon') {
                setPolygonReversed(geojson.coordinates[0]);
                setPolygon(reverseList(geojson.coordinates[0]));
                setCenterPoint([data.lat, data.lon]);
                setButtonDisabled(false);
                setNoData(false)
            }

            if (geojson.type === 'Point') {
                setCenterPoint(geojson.coordinates.reverse());
                setNoData(false)
            }
         }

        return getData();
    }


    return (
        <>
            <p>#everystreet is a running challenge in which you attempt to run every street within a chosen area.</p>
            <p>This app helps you run your neighbourhood
            in <a href="everystreet_algorithm.pdf" target="_blank" rel="noreferrer">the most optimal way</a>. Simply
            select or search for an area on the map, generate the route, and run #everystreet!</p>

            { loading && <RouteLoader progress={progress} /> }

            {!requestSent && <>
              <form className="search-form" onSubmit={(e) => getNeighbourhood(e)}>
                <input type="text" value={neighbourhood}  className='search' onChange={(e) => setNeighbourhood(e.target.value)} />
                <input type="submit" value="Search" className='submit' />
                {noData && <em> No such place has been found!</em>}
              </form>
              <p></p>
                <Map center={centerPoint} style={{ height: "50vh" }} animate={true} zoom={14}>
                    <TileLayer
                        attribution='&amp;copy <a href="https://osm.org/copyright">OpenStreetMap</a>'
                        url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
                    />
                    <FeatureGroup>
                        <EditControl
                            position='topright'
                            draw={{
                                polygon: true,
                                polyline: false,
                                circle: false,
                                marker: false,
                                rectangle: false,
                                circlemarker: false
                            }}
                            onCreated={ (e) => {
                                setPolygon(e.layer.toGeoJSON().geometry.coordinates[0]);
                                setPolygonReversed(e.layer.toGeoJSON().geometry.coordinates[0]);
                                setButtonDisabled(false)
                            }}
                            onEdited={ (e) => {
                                if (e.layers._layers.length) {
                                    setPolygon(e.layers._layers[Object.keys(e.layers._layers)].toGeoJSON().geometry.coordinates[0]);
                                    setPolygonReversed(e.layers._layers[Object.keys(e.layers._layers)].toGeoJSON().geometry.coordinates[0])
                                }
                            }}
                            onDeleted={ (e) => {
                                setPolygon([]);
                                setButtonDisabled(true)
                            }}
                        />
                        { polygon && <Polygon positions={polygon} key={Math.random()} /> }
                    </FeatureGroup>
                </Map>
                <Spacer type="normal" />
                <p></p>
                <button
                    disabled={buttonDisabled}
                    onClick={() => {
                        postRequest(polygonReversed, name);
                        setLoading(true);
                        setButtonDisabled(true);
                    }
                 }>Generate route</button>
            </>}
            {requestSent && <>
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

                    </ul>
                </>}
            </>}
        </>
    );
}
