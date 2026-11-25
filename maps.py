from __future__ import annotations

from html import escape
from typing import Iterable

import folium


SEVERITY_COLORS = {
    "Low": "green",
    "Medium": "orange",
    "High": "red",
}


def google_maps_link(latitude: float, longitude: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"


def google_maps_directions_link(
    source_latitude: float,
    source_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
) -> str:
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={source_latitude},{source_longitude}"
        f"&destination={destination_latitude},{destination_longitude}"
        "&travelmode=driving"
    )


def build_popup(record: dict) -> str:
    hazard_type = escape(str(record.get("hazard_type", "Hazard")))
    severity = escape(str(record.get("severity", "Low")))
    timestamp = escape(str(record.get("timestamp", "")))
    location_name = escape(str(record.get("location_name", "")))
    verification_status = escape(str(record.get("verification_status", "Pending")))
    lat = float(record.get("latitude", 0.0))
    lon = float(record.get("longitude", 0.0))
    return f"""
    <div style="width: 240px; font-family: Arial, sans-serif;">
        <h4 style="margin-bottom: 8px;">{hazard_type}</h4>
        <div><strong>Severity:</strong> {severity}</div>
        <div><strong>Date Reported:</strong> {timestamp}</div>
        <div><strong>Location:</strong> {location_name or f'{lat:.6f}, {lon:.6f}'}</div>
        <div><strong>Verification:</strong> {verification_status}</div>
        <div style="margin-top: 8px;">
            <a href="{google_maps_link(lat, lon)}" target="_blank">Open in Google Maps</a>
        </div>
    </div>
    """


def build_map_html(records: Iterable[dict], latitude: float, longitude: float) -> str:
    records = list(records)
    center = [latitude, longitude]
    if records:
        center = [float(records[0].get("latitude", latitude)), float(records[0].get("longitude", longitude))]

    hazard_map = folium.Map(location=center, zoom_start=13, tiles="CartoDB dark_matter")
    folium.TileLayer("OpenStreetMap").add_to(hazard_map)

    for record in records:
        severity = str(record.get("severity", "Low"))
        marker = folium.Marker(
            location=[float(record.get("latitude", latitude)), float(record.get("longitude", longitude))],
            popup=folium.Popup(build_popup(record), max_width=320),
            tooltip=f"{record.get('hazard_type', 'Hazard')} - {severity}",
            icon=folium.Icon(color=SEVERITY_COLORS.get(severity, "blue"), icon="exclamation-sign"),
        )
        marker.add_to(hazard_map)

    return hazard_map._repr_html_()
