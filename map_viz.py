"""
map_viz.py — folium map rendering for triangulator.

Depends on: pip install folium
"""

import math
import tempfile
import webbrowser

import folium

from geo import cone_polygon, point_at_bearing


_CAM_COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
BEARING_UNCERTAINTY_DEG = 5.0   # ± degrees shaded around each ray


def show_map(cameras, est_lat, est_lon,
             uncertainty_deg=BEARING_UNCERTAINTY_DEG):
    """
    Render an interactive folium map and open it in the browser.

    cameras   : list of camera dicts from geo.extract_camera
    est_lat   : estimated target latitude
    est_lon   : estimated target longitude
    """
    all_lats = [c["lat"] for c in cameras] + [est_lat]
    all_lons = [c["lon"] for c in cameras] + [est_lon]
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=center, tiles="OpenStreetMap")

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    for i, cam in enumerate(cameras):
        color = _CAM_COLORS[i % len(_CAM_COLORS)]
        clat, clon, bearing = cam["lat"], cam["lon"], cam["bearing"]

        # Extend ray 30 % past the estimated target (minimum 15 km)
        dist_to_target = math.hypot(
            (est_lat - clat) * 111_320,
            (est_lon - clon) * 111_320 * math.cos(math.radians(clat)),
        )
        ray_len = max(dist_to_target * 1.3, 15_000)

        # Shaded uncertainty cone
        folium.Polygon(
            locations=cone_polygon(clat, clon, bearing, uncertainty_deg, ray_len),
            color=color,
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=0.15,
        ).add_to(m)

        # Central bearing ray (dashed line)
        ray_end = point_at_bearing(clat, clon, bearing, ray_len)
        folium.PolyLine(
            locations=[[clat, clon], list(ray_end)],
            color=color,
            weight=2,
            opacity=0.85,
            dash_array="8 5",
        ).add_to(m)

        # Camera position marker
        popup_html = (
            f"<b>{cam['file']}</b><br>"
            f"{clat:.5f}°N, {abs(clon):.5f}°W<br>"
            f"Bearing: {bearing:.1f}°&nbsp;&nbsp;FOV: {cam['hfov']:.1f}°"
            + ("<br><i>front/selfie camera</i>" if cam["is_front"] else "")
        )
        folium.CircleMarker(
            location=[clat, clon],
            radius=9,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=cam["file"],
        ).add_to(m)

    # Estimated target marker
    folium.CircleMarker(
        location=[est_lat, est_lon],
        radius=11,
        color="white",
        weight=2,
        fill=True,
        fill_color="#F59E0B",
        fill_opacity=1.0,
        popup=folium.Popup(
            f"<b>Estimated position</b><br>"
            f"{est_lat:.6f}°N, {abs(est_lon):.6f}°W<br>"
            f"<a href='https://maps.google.com/?q={est_lat:.6f},{est_lon:.6f}'"
            f" target='_blank'>Open in Google Maps ↗</a>",
            max_width=280,
        ),
        tooltip="Estimated target",
    ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    # Fit viewport to all content
    pad = 0.02
    m.fit_bounds([
        [min(all_lats) - pad, min(all_lons) - pad],
        [max(all_lats) + pad, max(all_lons) + pad],
    ])

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        map_path = f.name
    m.save(map_path)
    webbrowser.open(f"file://{map_path}")
    print(f"  Map : file://{map_path}")
