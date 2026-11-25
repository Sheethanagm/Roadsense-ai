from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Iterable

from geopy.distance import geodesic


SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3}
SEVERITY_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def risk_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 60:
        return "orange"
    return "red"


def score_label(score: int) -> str:
    if score >= 80:
        return "Safe"
    if score >= 60:
        return "Moderate Risk"
    return "High Risk"


def point_to_segment_distance_km(point_a: tuple[float, float], point_b: tuple[float, float], point_p: tuple[float, float]) -> float:
    ax, ay = _project(point_a)
    bx, by = _project(point_b)
    px, py = _project(point_p)

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return sqrt(apx * apx + apy * apy) / 1000.0
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    dx = px - closest_x
    dy = py - closest_y
    return sqrt(dx * dx + dy * dy) / 1000.0


def _project(point: tuple[float, float]) -> tuple[float, float]:
    lat, lon = point
    radius = 6371000.0
    x = radians(lon) * radius * cos(radians(lat))
    y = radians(lat) * radius
    return x, y


def infer_severity(hazard: dict) -> str:
    return str(hazard.get("severity") or "Low")


def summarize_hazards(records: Iterable[dict]) -> dict:
    rows = list(records)
    total = len(rows)
    severity_counts = Counter(infer_severity(item) for item in rows)
    type_counts = Counter(str(item.get("hazard_type", "Hazard")) for item in rows)
    active_high = sum(1 for item in rows if infer_severity(item) == "High" and str(item.get("verification_status", "")).lower() != "resolved")
    dates = [str(item.get("timestamp", "")) for item in rows if item.get("timestamp")]
    last_report = max(dates) if dates else ""
    return {
        "total": total,
        "severity_counts": dict(severity_counts),
        "type_counts": dict(type_counts),
        "active_high": active_high,
        "last_report": last_report,
    }


def nearby_hazard_count(records: Iterable[dict], latitude: float, longitude: float, radius_km: float = 5.0) -> int:
    origin = (latitude, longitude)
    count = 0
    for record in records:
        hazard_point = (safe_float(record.get("latitude")), safe_float(record.get("longitude")))
        if geodesic(origin, hazard_point).km <= radius_km:
            count += 1
    return count


def filter_route_hazards(records: Iterable[dict], source: tuple[float, float], destination: tuple[float, float], corridor_km: float = 1.5) -> list[dict]:
    route_hazards = []
    for record in records:
        point = (safe_float(record.get("latitude")), safe_float(record.get("longitude")))
        distance_km = point_to_segment_distance_km(source, destination, point)
        if distance_km <= corridor_km:
            enriched = dict(record)
            enriched["distance_to_route_km"] = distance_km
            route_hazards.append(enriched)
    return sorted(route_hazards, key=lambda item: (safe_float(item.get("distance_to_route_km")), item.get("timestamp", "")))


def build_route_plan(records: Iterable[dict], source: tuple[float, float], destination: tuple[float, float]) -> dict:
    hazards = filter_route_hazards(records, source, destination)
    total_hazards = len(hazards)
    severity_breakdown = Counter(infer_severity(item) for item in hazards)
    hazard_categories = Counter(str(item.get("hazard_type", "Hazard")) for item in hazards)

    base_distance = geodesic(source, destination).km
    added_distance = min(0.35 * base_distance, 2.5) if total_hazards else 0.0
    fastest_distance = round(base_distance, 2)
    safest_distance = round(base_distance + added_distance, 2)

    raw_score = 100 - (total_hazards * 8) - (severity_breakdown.get("High", 0) * 12) - (severity_breakdown.get("Medium", 0) * 5)
    safety_score = max(0, min(100, int(raw_score)))
    route_risk_score = max(0, min(100, 100 - safety_score))

    safest_option = "Safest Route" if safety_score >= 60 else "Fastest Route with Caution"
    comparison = [
        {
            "route": "Fastest Route",
            "distance_km": fastest_distance,
            "risk_score": min(100, route_risk_score + 12),
            "safety_score": max(0, safety_score - 12),
        },
        {
            "route": "Safest Route",
            "distance_km": safest_distance,
            "risk_score": route_risk_score,
            "safety_score": safety_score,
        },
    ]

    high_risk_segments = [
        {
            "hazard_type": item.get("hazard_type", "Hazard"),
            "severity": infer_severity(item),
            "location_name": item.get("location_name", ""),
            "distance_to_route_km": round(safe_float(item.get("distance_to_route_km")), 2),
        }
        for item in hazards
        if infer_severity(item) == "High"
    ][:5]

    return {
        "total_hazards": total_hazards,
        "hazard_categories": dict(hazard_categories),
        "severity_breakdown": dict(severity_breakdown),
        "safety_score": safety_score,
        "route_risk_score": route_risk_score,
        "comparison": comparison,
        "high_risk_segments": high_risk_segments,
        "recommended_route": safest_option,
        "fastest_distance_km": fastest_distance,
        "safest_distance_km": safest_distance,
    }


def trend_counts(records: Iterable[dict], frequency: str = "D") -> list[dict]:
    buckets: dict[str, int] = {}
    for record in records:
        ts = str(record.get("timestamp", "")).strip()
        if not ts:
            continue
        try:
            stamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if frequency == "W":
            key = f"{stamp.isocalendar().year}-W{stamp.isocalendar().week:02d}"
        elif frequency == "M":
            key = stamp.strftime("%Y-%m")
        else:
            key = stamp.strftime("%Y-%m-%d")
        buckets[key] = buckets.get(key, 0) + 1
    return [{"period": key, "count": buckets[key]} for key in sorted(buckets)]

