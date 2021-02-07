GPX_HEADERS = {'Content-type': 'text/xml'}

TRACE_POINT = '<trkpt lat="{lat}" lon="{lon}"><ele>{id}</ele><time>{timestamp}</time></trkpt>'

TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<gpx version="1.1"
    creator="EMTAC BTGPS Trine II DataLog Dump 1.0 - http://www.ayeltd.biz"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns="http://www.topografix.com/GPX/1/1"
    xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
    <name>{name}</name>
    <wpt lat="{center_lat}" lon="{center_lon}">
        <ele>2372</ele>
        <name>{name}</name>
    </wpt>
    <trk>
        <name>{name}</name>
        <number>1</number>
        <trkseg>
            {trace_points}
        </trkseg>
    </trk>
</gpx>
"""
