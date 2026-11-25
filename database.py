from __future__ import annotations

from detect import classify_severity, normalize_label, recommendation_for


def alert_message(hazard_type: str, severity: str) -> str:
    label = normalize_label(hazard_type)
    if severity == "High":
        return f"Large {label} detected ahead. Reduce speed immediately."
    if severity == "Medium":
        return f"{label} detected ahead. Slow down and stay alert."
    return f"{label} detected. Proceed with caution."


def build_alert_payload(detection: dict) -> dict:
    hazard_type = detection.get("hazard_type", "Hazard")
    confidence = float(detection.get("confidence", 0.0))
    severity = detection.get("severity") or classify_severity(hazard_type, confidence)
    recommendation = detection.get("recommendation") or recommendation_for(hazard_type, severity)
    return {
        "hazard_type": normalize_label(hazard_type),
        "confidence": confidence,
        "severity": severity,
        "recommendation": recommendation,
        "message": alert_message(hazard_type, severity),
    }
