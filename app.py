__future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from alerts import build_alert_payload
from database import HazardDatabase
from detect import DetectionEngine, load_image_file, process_video_file, save_image
from gps import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, geocode_location, resolve_location
from maps import build_map_html, google_maps_directions_link, google_maps_link
from safety import build_route_plan, nearby_hazard_count, risk_color, score_label, summarize_hazards


APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / "uploads"
OUTPUTS_DIR = APP_DIR / "outputs"
MODEL_PATH = APP_DIR / "model" / "best.pt"
DATABASE_PATH = APP_DIR / "data" / "hazards.db"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Road Safety Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(16, 185, 129, 0.08), transparent 28%),
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.08), transparent 24%),
                linear-gradient(180deg, #07111f 0%, #0f172a 100%);
            color: #e5eef8;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1324 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        .hero-card, .metric-card, .panel-card, .route-card, .alert-card {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.22);
            padding: 1rem 1.1rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
            color: #f8fafc;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.5;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: #f8fafc;
            margin-top: 0.2rem;
        }
        .metric-label {
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.73rem;
        }
        .small-note {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        .tag {
            display: inline-block;
            padding: 0.3rem 0.6rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.14);
            border: 1px solid rgba(59,130,246,0.30);
            color: #bfdbfe;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
            font-size: 0.83rem;
        }
        .status-safe { color: #22c55e; font-weight: 700; }
        .status-moderate { color: #f59e0b; font-weight: 700; }
        .status-high { color: #ef4444; font-weight: 700; }
        .alert-banner {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            color: white;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.12);
            margin: 0.35rem 0 1rem 0;
        }
        .emergency-button a {
            display: inline-block;
            padding: 0.8rem 1rem;
            border-radius: 14px;
            background: linear-gradient(90deg, #dc2626, #ef4444);
            color: white !important;
            font-weight: 700;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_detector() -> DetectionEngine:
    return DetectionEngine(model_path=MODEL_PATH)


@st.cache_resource
def get_database() -> HazardDatabase:
    return HazardDatabase(DATABASE_PATH)


def init_state() -> None:
    defaults = {
        "page": "Home",
        "current_location": {"lat": DEFAULT_LATITUDE, "lon": DEFAULT_LONGITUDE, "label": "Chennai, India"},
        "report_location": {"lat": DEFAULT_LATITUDE, "lon": DEFAULT_LONGITUDE, "label": "Chennai, India"},
        "route_source": {"lat": DEFAULT_LATITUDE, "lon": DEFAULT_LONGITUDE, "label": "Source location"},
        "route_destination": {"lat": DEFAULT_LATITUDE + 0.08, "lon": DEFAULT_LONGITUDE + 0.08, "label": "Destination location"},
        "reverse_geocoded": "",
        "report_status": "Pending",
        "driver_voice_message": "",
        "driver_visual_message": "",
        "driver_severity": "Low",
        "default_emergency": "112",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_sample_location() -> None:
    st.session_state.current_location = {"lat": DEFAULT_LATITUDE, "lon": DEFAULT_LONGITUDE, "label": "Chennai, India"}
    st.session_state.report_location = {"lat": DEFAULT_LATITUDE, "lon": DEFAULT_LONGITUDE, "label": "Chennai, India"}
    st.session_state.reverse_geocoded = ""


def metric_card(label: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def location_picker(
    title: str,
    state_key: str,
    default_lat: float,
    default_lon: float,
    default_label: str,
    *,
    sidebar: bool = False,
    height: int = 280,
    search_placeholder: str = "Search a place",
) -> dict:
    ui = st.sidebar if sidebar else st
    current = st.session_state.get(
        state_key,
        {"lat": default_lat, "lon": default_lon, "label": default_label},
    )

    ui.markdown(f"#### {title}")
    search_value = ui.text_input(search_placeholder, key=f"{state_key}_search")
    search_cols = ui.columns([1, 1])
    if search_cols[0].button("Find location", key=f"{state_key}_find"):
        lat, lon, label = geocode_location(search_value)
        if label:
            current = {"lat": lat, "lon": lon, "label": label}
            st.session_state[state_key] = current
    if search_cols[1].button("Use sample", key=f"{state_key}_sample"):
        current = {"lat": default_lat, "lon": default_lon, "label": default_label}
        st.session_state[state_key] = current

    map_view = folium.Map(location=[current["lat"], current["lon"]], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        [current["lat"], current["lon"]],
        tooltip=current.get("label", title),
        popup=current.get("label", title),
    ).add_to(map_view)
    folium.LatLngPopup().add_to(map_view)
    map_data = st_folium(map_view, key=f"{state_key}_map", height=height, use_container_width=True)

    if map_data and map_data.get("last_clicked"):
        clicked_lat = float(map_data["last_clicked"]["lat"])
        clicked_lon = float(map_data["last_clicked"]["lng"])
        clicked_label = resolve_location(clicked_lat, clicked_lon) or f"{clicked_lat:.6f}, {clicked_lon:.6f}"
        previous = st.session_state.get(state_key, {})
        if abs(previous.get("lat", 0.0) - clicked_lat) > 1e-6 or abs(previous.get("lon", 0.0) - clicked_lon) > 1e-6:
            current = {"lat": clicked_lat, "lon": clicked_lon, "label": clicked_label}
            st.session_state[state_key] = current
            st.rerun()
    else:
        st.session_state[state_key] = current

    ui.caption(current.get("label", default_label))
    ui.markdown(f"[Open in Google Maps]({google_maps_link(current['lat'], current['lon'])})")
    return st.session_state[state_key]


def nav_label_map() -> dict[str, str]:
    return {
        "Home": "Home",
        "Driver Mode": "Driver Mode",
        "Report Hazard": "Report Hazard",
        "Hazard Map": "Hazard Map",
        "Route Safety": "Route Safety",
        "Analytics": "Analytics",
        "Settings": "Settings",
    }


def sidebar_controls() -> tuple[str, float, float, str]:
    st.sidebar.markdown("## Road Safety Platform")
    st.sidebar.caption("Safer travel for drivers, commuters, and public road users")
    labels = list(nav_label_map().keys())
    selected_label = st.sidebar.radio("Navigation", labels, index=labels.index(next(k for k, v in nav_label_map().items() if v == st.session_state.page)))
    st.session_state.page = nav_label_map()[selected_label]

    st.sidebar.markdown("---")
    current = location_picker(
        "Current Area",
        "current_location",
        DEFAULT_LATITUDE,
        DEFAULT_LONGITUDE,
        "Chennai, India",
        sidebar=True,
        height=210,
        search_placeholder="Search current area",
    )
    latitude = current["lat"]
    longitude = current["lon"]
    location_name = current["label"]

    st.sidebar.markdown("---")
    summary = get_database().fetch_summary()
    total = summary["total_reports"]
    active = summary["active_high_risk"]
    st.sidebar.markdown("### Road Safety Status")
    st.sidebar.success(f"{total} reports tracked")
    st.sidebar.warning(f"{active} active high-risk points")
    return st.session_state.page, latitude, longitude, location_name


def js_voice_alert(message: str) -> None:
    safe_message = message.replace("\\", "\\\\").replace("`", "").replace('"', "'")
    st.components.v1.html(
        f"""
        <script>
        const speak = () => {{
            const utterance = new SpeechSynthesisUtterance("{safe_message}");
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }};
        speak();
        </script>
        """,
        height=0,
    )


def format_status(score: int) -> str:
    label = score_label(score)
    return f"<span class='status-{risk_color(score)}'>{label}</span>"


def home_page(db: HazardDatabase, latitude: float, longitude: float) -> None:
    summary = db.fetch_summary()
    recent = db.fetch_recent_alerts(limit=6)
    all_reports = db.fetch_map_records()
    nearby_count = nearby_hazard_count(all_reports, latitude, longitude, radius_km=5.0)
    high_risk_nearby = sum(
        1
        for row in all_reports
        if row["severity"] == "High"
        and nearby_hazard_count([row], latitude, longitude, radius_km=10.0) > 0
    )
    safety_score = max(35, 100 - nearby_count * 5 - high_risk_nearby * 12)
    weather_alerts = build_weather_alerts(all_reports)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Road Safety Status</div>
            <div class="hero-subtitle">
                Stay aware of nearby hazards, safer routes, and active alerts before and during travel.
            </div>
            <div style="margin-top: 0.5rem;">
                <span class="tag">Safer travel</span>
                <span class="tag">Live alerts</span>
                <span class="tag">Route awareness</span>
                <span class="tag">Community reporting</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Nearby Hazards", str(nearby_count), "Within your selected area")
    with c2:
        metric_card("Safe Routes Available", "Yes" if safety_score >= 60 else "Review", "Route safety suggestions ready")
    with c3:
        metric_card("Active Alerts", str(summary["active_high_risk"]), "High-risk points needing attention")
    with c4:
        metric_card("Last Reported Hazard", summary["last_reported"] or "No recent reports", "Latest community update")

    left, right = st.columns([1.45, 1])
    with left:
        section_title("Road Safety Overview")
        overview_cols = st.columns(3)
        with overview_cols[0]:
            metric_card("Road Safety Score", f"{safety_score}/100", "Higher is safer")
        with overview_cols[1]:
            metric_card("High Risk Areas Nearby", str(high_risk_nearby), "Requires caution")
        with overview_cols[2]:
            metric_card("Recent Alerts", str(len(recent)), "Latest safety notices")

        st.markdown(
            f"""
            <div class="alert-card">
                <strong>Safety Status:</strong> {format_status(safety_score)}
                <div class="small-note">Score considers nearby reports and active high-risk points.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_title("Recent Alerts")
        if recent:
            alert_rows = []
            for row in recent:
                alert_rows.append(
                    {
                        "Hazard": row["hazard_type"],
                        "Severity": row["severity"],
                        "Location": row.get("location_name") or f'{row["latitude"]:.4f}, {row["longitude"]:.4f}',
                        "Status": row["verification_status"],
                        "Time": row["timestamp"],
                    }
                )
            st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No recent alerts yet. Reports will appear here as they are added.")

    with right:
        section_title("Weather Impact Alerts")
        if weather_alerts:
            for alert in weather_alerts[:4]:
                st.markdown(f"<div class='panel-card'>{alert}</div>", unsafe_allow_html=True)
                st.write("")
        else:
            st.info("No weather-sensitive impacts are currently reported nearby.")

        section_title("Emergency Assistance")
        st.markdown(
            f"""
            <div class="emergency-button">
                <a href="tel:{st.session_state.default_emergency}">Call Emergency Assistance</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_weather_alerts(records: list[dict]) -> list[str]:
    hazards = {str(row.get("hazard_type", "")).lower() for row in records}
    alerts = []
    if "waterlogging" in hazards:
        alerts.append("Wet-road conditions reported nearby. Brake gently and keep extra distance.")
    if "fallen tree" in hazards:
        alerts.append("Storm-related obstruction reported nearby. Expect delays and follow diversions.")
    if "road debris" in hazards:
        alerts.append("Loose debris reported on or near the road. Stay alert for sudden lane changes.")
    if "animal" in hazards:
        alerts.append("Animal crossing reports nearby. Reduce speed in open-road stretches.")
    return alerts


def infer_warning_message(detection: dict, distance_m: int | None = None) -> tuple[str, str, str]:
    payload = build_alert_payload(detection)
    severity = payload["severity"]
    message = payload["message"]
    recommendation = payload["recommendation"]
    if distance_m is not None:
        recommendation = f"{recommendation} Approx. {distance_m} m ahead."
    return message, severity, recommendation


def estimate_distance_m(bbox: list[int], frame_shape: tuple[int, int, int]) -> int:
    frame_area = frame_shape[0] * frame_shape[1]
    box_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    ratio = min(1.0, box_area / frame_area)
    estimated = int(100 - (ratio * 90))
    return max(5, estimated)


def save_reports(
    db: HazardDatabase,
    detections: list[dict],
    source_type: str,
    latitude: float,
    longitude: float,
    location_name: str,
    reporter_id: str,
    reporter_name: str,
    reporter_contact: str,
    media_path: str,
    verification_status: str,
    notes: str = "",
) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    if detections:
        for detection in detections:
            db.insert_report(
                {
                    "hazard_type": detection["hazard_type"],
                    "severity": detection["severity"],
                    "latitude": latitude,
                    "longitude": longitude,
                    "location_name": location_name,
                    "timestamp": timestamp,
                    "media_path": media_path,
                    "source_type": source_type,
                    "verification_status": verification_status,
                    "bbox_data": str(detection.get("bbox", "")),
                    "reporter_id": reporter_id,
                    "reporter_name": reporter_name,
                    "reporter_contact": reporter_contact,
                    "notes": notes,
                }
            )


def home_top_metrics(db: HazardDatabase, latitude: float, longitude: float) -> None:
    reports = db.fetch_map_records()
    nearby = nearby_hazard_count(reports, latitude, longitude, radius_km=5.0)
    active_high = sum(1 for row in reports if row["severity"] == "High" and row.get("is_active", 1))
    total = len(reports)
    last = db.fetch_summary()["last_reported"] or "No report yet"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Nearby Hazards", str(nearby), "Close to your current area")
    with c2:
        metric_card("Safe Routes Available", "Yes" if nearby < 8 else "Review", "Safer option available")
    with c3:
        metric_card("Active Alerts", str(active_high), "Needs caution")
    with c4:
        metric_card("Last Reported Hazard", last, "Latest update from the road")


def driver_mode_page(detector: DetectionEngine, db: HazardDatabase, latitude: float, longitude: float, location_name: str) -> None:
    st.markdown("## Driver Mode")
    st.caption("Real-time hazard prevention while travelling")
    st.info("Start monitoring before you move. The screen is optimized for quick glances and immediate warnings.")
    compact_mode = st.toggle("Compact view", value=True)

    try:
        from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
    except Exception as exc:
        st.error("Live monitoring is unavailable right now.")
        st.code(str(exc))
        return

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class VideoProcessor:
        def __init__(self) -> None:
            self.latest_warning = ""
            self.latest_severity = "Low"
            self.latest_recommendation = ""
            self.latest_distance = None
            self.latest_frame_has_alert = False

        def recv(self, frame):
            import av

            image = frame.to_ndarray(format="bgr24")
            annotated, detections = detector.detect_image(image, 0.25)
            self.latest_frame_has_alert = bool(detections)
            if detections:
                top = max(detections, key=lambda item: item["confidence"])
                distance = estimate_distance_m(top["bbox"], image.shape)
                message, severity, recommendation = infer_warning_message(top, distance)
                self.latest_warning = message
                self.latest_severity = severity
                self.latest_recommendation = recommendation
                self.latest_distance = distance
                color = (0, 0, 255) if severity == "High" else (0, 165, 255) if severity == "Medium" else (0, 200, 0)
                cv2.putText(annotated, message[:70], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.putText(annotated, recommendation[:72], (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="driver-mode",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if ctx.video_processor and ctx.video_processor.latest_warning:
        severity = ctx.video_processor.latest_severity
        color = "#dc2626" if severity == "High" else "#f59e0b" if severity == "Medium" else "#16a34a"
        st.markdown(
            f"""
            <div class="alert-banner" style="background: {color};">
                {ctx.video_processor.latest_warning}<br/>
                {ctx.video_processor.latest_recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )
        js_voice_alert(ctx.video_processor.latest_warning)
    else:
        st.markdown(
            """
            <div class="panel-card">
                Live monitoring is ready. Start the camera to receive visual warnings and spoken alerts.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if compact_mode:
        st.caption("Compact mode keeps the warning view centered for quick mobile-style checks.")


def report_hazard_page(detector: DetectionEngine, db: HazardDatabase, latitude: float, longitude: float, location_name: str) -> None:
    st.markdown("## Report Hazard")
    st.caption("Upload a photo, confirm the hazard, and share a safer road report.")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    reporter_col1, reporter_col2, reporter_col3 = st.columns(3)
    with reporter_col1:
        reporter_name = st.text_input("Your name", value=st.session_state.get("reporter_name", ""))
    with reporter_col2:
        reporter_contact = st.text_input("Phone or email", value=st.session_state.get("reporter_contact", ""))
    with reporter_col3:
        reporter_id = st.text_input("Reporter ID", value=st.session_state.get("reporter_id", ""))

    selected_location = location_picker(
        "Hazard Location",
        "report_location",
        latitude,
        longitude,
        location_name,
        height=300,
        search_placeholder="Search hazard location",
    )
    notes = st.text_area("Notes", placeholder="Add details for other road users or authorities.")

    suggested_type = "Manual report"
    suggested_severity = "Medium"
    detected_detections: list[dict] = []
    preview_image = None

    if uploaded is not None:
        image_path = UPLOADS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded.name}"
        image_path.write_bytes(uploaded.getbuffer())
        preview_image = load_image_file(image_path)
        annotated, detected_detections = detector.detect_image(preview_image, 0.25)
        save_image(OUTPUTS_DIR / f"report_{image_path.stem}.png", annotated)

        left, right = st.columns(2)
        with left:
            st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)
        with right:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Marked image", use_container_width=True)

        if detected_detections:
            top = max(detected_detections, key=lambda item: item["confidence"])
            suggested_type = top["hazard_type"]
            suggested_severity = top["severity"]
            st.success(f"Suggested report: {suggested_type} ({suggested_severity})")
        else:
            st.warning("No clear issue was identified. You can still create a manual report.")
    else:
        st.info("Upload a photo to let the system suggest a report automatically.")

    if detected_detections:
        hazard_type = st.selectbox(
            "Hazard type",
            options=[
                suggested_type,
                "Pothole",
                "Road Crack",
                "Waterlogging",
                "Fallen Tree",
                "Road Debris",
                "Animal",
                "Damaged Vehicle",
                "Accident Zone",
            ],
            index=0,
        )
    else:
        hazard_type = st.selectbox(
            "Hazard type",
            options=["Pothole", "Road Crack", "Waterlogging", "Fallen Tree", "Road Debris", "Animal", "Damaged Vehicle", "Accident Zone"],
        )

    severity = st.selectbox("Severity", options=["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(suggested_severity if detected_detections else "Medium"))
    verification_status = st.selectbox("Verification status", options=["Pending", "Verified", "Needs Review"], index=1 if detected_detections else 0)

    confirm = st.button("Confirm report", type="primary")
    if confirm:
        media_path = ""
        if uploaded is not None:
            media_path = str(image_path)
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        db.insert_report(
            {
                "hazard_type": hazard_type,
                "severity": severity,
                "latitude": selected_location["lat"],
                "longitude": selected_location["lon"],
                "location_name": selected_location["label"],
                "timestamp": timestamp,
                "media_path": media_path,
                "source_type": "User Report",
                "verification_status": verification_status,
                "bbox_data": str(detected_detections[0].get("bbox", "")) if detected_detections else "",
                "reporter_id": reporter_id,
                "reporter_name": reporter_name,
                "reporter_contact": reporter_contact,
                "notes": notes,
            }
        )
        st.success("Your hazard report has been saved.")


def hazard_map_page(db: HazardDatabase) -> None:
    st.markdown("## Hazard Map")
    st.caption("Explore reported hazards, filter by risk, and inspect danger zones.")
    reports = db.fetch_map_records()
    if not reports:
        st.info("No hazards have been reported yet.")
        return

    filters = st.columns(4)
    severity_filter = filters[0].selectbox("Severity", ["All", "Low", "Medium", "High"])
    hazard_types = ["All"] + sorted({row["hazard_type"] for row in reports})
    hazard_filter = filters[1].selectbox("Hazard type", hazard_types)
    start_date = filters[2].date_input("From", value=datetime.utcnow().date() - timedelta(days=30))
    end_date = filters[3].date_input("To", value=datetime.utcnow().date())

    filtered = db.fetch_map_records(
        severity=severity_filter,
        hazard_type=hazard_filter,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    if not filtered:
        st.warning("No hazards match the selected filters.")
        return

    st.components.v1.html(build_map_html(filtered, DEFAULT_LATITUDE, DEFAULT_LONGITUDE), height=680, scrolling=True)

    st.markdown("### Report Details")
    table_rows = [
        {
            "Hazard Type": row["hazard_type"],
            "Severity": row["severity"],
            "Date Reported": row["timestamp"],
            "Location": row.get("location_name") or f'{row["latitude"]:.4f}, {row["longitude"]:.4f}',
            "Verification": row["verification_status"],
        }
        for row in filtered[:20]
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    try:
        from folium.plugins import HeatMap
        import folium

        heat_center = [filtered[0]["latitude"], filtered[0]["longitude"]]
        danger_map = folium.Map(location=heat_center, zoom_start=12, tiles="OpenStreetMap")
        HeatMap([[row["latitude"], row["longitude"]] for row in filtered]).add_to(danger_map)
        st.markdown("### Danger Zone Heatmap")
        st.components.v1.html(danger_map._repr_html_(), height=520, scrolling=True)
    except Exception:
        st.info("Heatmap view is unavailable right now.")


def route_safety_page(db: HazardDatabase) -> None:
    st.markdown("## Route Safety Checker")
    st.caption("Compare travel options and choose the safer path.")
    st.info("Pick both locations on the map or search for a place name. The app will calculate a safety summary automatically.")
    left_picker, right_picker = st.columns(2)
    with left_picker:
        source = location_picker(
            "Source",
            "route_source",
            DEFAULT_LATITUDE,
            DEFAULT_LONGITUDE,
            "Source location",
            height=260,
            search_placeholder="Search source",
        )
    with right_picker:
        destination = location_picker(
            "Destination",
            "route_destination",
            DEFAULT_LATITUDE + 0.08,
            DEFAULT_LONGITUDE + 0.08,
            "Destination location",
            height=260,
            search_placeholder="Search destination",
        )

    calculate = st.button("Generate route safety report", type="primary")
    source_lat, source_lon, source_address = source["lat"], source["lon"], source["label"]
    destination_lat, destination_lon, destination_address = destination["lat"], destination["lon"], destination["label"]
    reports = db.fetch_map_records(active_only=True)
    route_plan = build_route_plan(reports, (source_lat, source_lon), (destination_lat, destination_lon))
    if calculate:
        st.success("Route safety report generated.")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f"**Source:** {source_address}")
        st.markdown(f"**Destination:** {destination_address}")
        st.write("")
        st.markdown(
            f"""
            <div class="panel-card">
                <strong>Route Safety Score:</strong> <span class="status-{risk_color(route_plan['safety_score'])}">{route_plan['safety_score']}/100</span><br/>
                <strong>Recommendation:</strong> {route_plan['recommended_route']}<br/>
                <strong>Total Hazards Along Route:</strong> {route_plan['total_hazards']}<br/>
                <strong>Fastest Route:</strong> {route_plan['fastest_distance_km']} km<br/>
                <strong>Safest Route:</strong> {route_plan['safest_distance_km']} km
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            **Source:** {source_address}<br/>
            **Destination:** {destination_address}
            """,
            unsafe_allow_html=True,
        )
        comparison_df = pd.DataFrame(route_plan["comparison"])
        comparison_df["Safety"] = comparison_df["safety_score"].apply(lambda score: f"{score}/100")
        comparison_df["Risk"] = comparison_df["risk_score"].apply(lambda score: f"{score}/100")
        st.dataframe(comparison_df[["route", "distance_km", "Safety", "Risk"]], use_container_width=True, hide_index=True)

        if route_plan["high_risk_segments"]:
            st.markdown("### High Risk Segments")
            st.dataframe(pd.DataFrame(route_plan["high_risk_segments"]), use_container_width=True, hide_index=True)
        else:
            st.success("No high-risk segments detected on this route.")

    with right:
        st.markdown("### Hazard Categories")
        if route_plan["hazard_categories"]:
            st.bar_chart(pd.Series(route_plan["hazard_categories"]))
        else:
            st.info("No route hazards found.")

        st.markdown("### Severity Breakdown")
        if route_plan["severity_breakdown"]:
            st.bar_chart(pd.Series(route_plan["severity_breakdown"]))
        else:
            st.info("No severity data available.")

    route_map = create_route_map(source_lat, source_lon, destination_lat, destination_lon, reports)
    st.markdown("### Route Preview")
    st.components.v1.html(route_map, height=560, scrolling=True)
    st.markdown(
        f"[Open directions in Google Maps]({google_maps_directions_link(source_lat, source_lon, destination_lat, destination_lon)})"
    )


def create_route_map(source_lat: float, source_lon: float, destination_lat: float, destination_lon: float, reports: list[dict]) -> str:
    import folium

    m = folium.Map(location=[source_lat, source_lon], zoom_start=12, tiles="OpenStreetMap")
    folium.Marker([source_lat, source_lon], tooltip="Source", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([destination_lat, destination_lon], tooltip="Destination", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([(source_lat, source_lon), (destination_lat, destination_lon)], color="#60a5fa", weight=4, opacity=0.8).add_to(m)

    for row in reports[:150]:
        color = "red" if row["severity"] == "High" else "orange" if row["severity"] == "Medium" else "green"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{row['hazard_type']} - {row['severity']}",
        ).add_to(m)
    return m._repr_html_()


def analytics_page(db: HazardDatabase) -> None:
    st.markdown("## Analytics")
    st.caption("Understand hazard trends, high-risk patterns, and report activity.")
    reports = db.fetch_map_records()
    summary = summarize_hazards(reports)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Hazard Reports", str(summary["total"]), "All reported road hazards")
    with c2:
        metric_card("Active High-Risk Hazards", str(summary["active_high"]), "Requires caution")
    with c3:
        metric_card("Most Common Type", max(summary["type_counts"], key=summary["type_counts"].get) if summary["type_counts"] else "N/A", "Most frequently reported")
    with c4:
        metric_card("Last Reported", summary["last_report"] or "No recent report", "Latest update")

    left, right = st.columns(2)
    with left:
        section_title("Most Common Hazard Types")
        if summary["type_counts"]:
            st.bar_chart(pd.Series(summary["type_counts"]).sort_values(ascending=False))
        else:
            st.info("No report data yet.")
    with right:
        section_title("Severity Distribution")
        if summary["severity_counts"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(summary["severity_counts"].values(), labels=summary["severity_counts"].keys(), autopct="%1.0f%%", startangle=90)
            ax.set_title("Severity Mix")
            st.pyplot(fig)
        else:
            st.info("No severity data yet.")

    st.markdown("### Daily Reports")
    daily = pd.DataFrame(db.fetch_trend_rows("D"))
    if not daily.empty:
        st.line_chart(daily.set_index("period")["count"])
    else:
        st.info("No daily trend data available.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Weekly Reports")
        weekly = pd.DataFrame(db.fetch_trend_rows("W"))
        if not weekly.empty:
            st.bar_chart(weekly.set_index("period")["count"])
        else:
            st.info("No weekly trend data available.")
    with c2:
        st.markdown("### Monthly Reports")
        monthly = pd.DataFrame(db.fetch_trend_rows("M"))
        if not monthly.empty:
            st.bar_chart(monthly.set_index("period")["count"])
        else:
            st.info("No monthly trend data available.")

    st.markdown("### Recent Reports")
    recent = db.fetch_reports(limit=25)
    if recent:
        table = pd.DataFrame(
            [
                {
                    "Hazard": row["hazard_type"],
                    "Severity": row["severity"],
                    "Location": row.get("location_name") or f'{row["latitude"]:.4f}, {row["longitude"]:.4f}',
                    "Status": row["verification_status"],
                    "Time": row["timestamp"],
                }
                for row in recent
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No reports to show yet.")


def settings_page(db: HazardDatabase) -> None:
    st.markdown("## Settings")
    st.caption("Adjust how the safety platform behaves for your trips.")
    st.session_state.default_emergency = st.text_input("Emergency assistance number", value=st.session_state.default_emergency)
    preferred_area = st.text_input("Preferred area label", value=st.session_state.current_location["label"])
    current_location = st.session_state.current_location
    current_location["label"] = preferred_area
    st.session_state.current_location = current_location
    st.checkbox("Allow spoken warnings", value=True, key="spoken_warnings")
    st.checkbox("Show compact driver view", value=True, key="compact_view")
    st.checkbox("Auto-refresh reports", value=False, key="auto_refresh")
    st.info("Crowd reporting, accident prediction, smart city links, and push notifications are ready for future expansion.")


def render_page(detector: DetectionEngine, db: HazardDatabase, page: str, latitude: float, longitude: float, location_name: str) -> None:
    if page == "Home":
        home_page(db, latitude, longitude)
    elif page == "Driver Mode":
        driver_mode_page(detector, db, latitude, longitude, location_name)
    elif page == "Report Hazard":
        report_hazard_page(detector, db, latitude, longitude, location_name)
    elif page == "Hazard Map":
        hazard_map_page(db)
    elif page == "Route Safety":
        route_safety_page(db)
    elif page == "Analytics":
        analytics_page(db)
    elif page == "Settings":
        settings_page(db)


def main() -> None:
    inject_css()
    init_state()
    detector = get_detector()
    db = get_database()
    page, latitude, longitude, location_name = sidebar_controls()

    st.caption(location_name)
    render_page(detector, db, page, latitude, longitude, location_name)


if __name__ == "__main__":
    main()
