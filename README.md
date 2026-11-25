# RoadSense AI

AI-powered road safety web app that detects road hazards with YOLOv8, maps community reports, and scores route safety.

## Features

- **Home:** safety score for your area, nearby hazard counts, recent alerts, and an emergency call button
- **Driver Mode:** live camera hazard detection with on-screen and spoken warnings
- **Report Hazard:** upload a photo, run detection, pick the location on a map, and save the report
- **Hazard Map:** interactive map with severity-coloured markers and filters
- **Route Safety:** pick a source and destination to get a safety score and a Google Maps directions link
- **Analytics:** hazard trends by day, week, or month, plus breakdowns by type and severity
- **Settings:** emergency number and display preferences

## Detected Hazards

Pothole, Road Crack, Waterlogging, Fallen Tree, Road Debris, Animal, Damaged Vehicle, Accident Zone

Each detection gets a severity (Low, Medium, or High) and a driving recommendation.

## Tech Stack

- Python 3.11
- Streamlit (web interface)
- YOLOv8 / Ultralytics and OpenCV (hazard detection)
- SQLite (report storage)
- Folium and streamlit-folium (maps)
- geopy (geocoding via Google Maps or OpenStreetMap Nominatim)
- streamlit-webrtc (live camera)

## Project Structure

```text
roadsense-ai/
├── app.py            # Streamlit UI and pages
├── main.py           # Launcher
├── detect.py         # YOLOv8 detection and severity logic
├── database.py       # SQLite storage
├── safety.py         # Route scoring and trend calculations
├── alerts.py         # Warning messages
├── gps.py            # Geocoding
├── maps.py           # Map builders
├── requirements.txt
└── model/
    └── best.pt       # Your trained model (not included)
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/roadsense-ai.git
cd roadsense-ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add a trained YOLOv8 model at `model/best.pt`. The model file is not included in this repository. The app runs without it, but hazard detection will not work.

4. Optional: set a Google Maps API key for better location search.

Windows:
```bash
set GOOGLE_MAPS_API_KEY=your_key_here
```

Mac/Linux:
```bash
export GOOGLE_MAPS_API_KEY=your_key_here
```

Without a key, the app uses OpenStreetMap Nominatim.

## Run

```bash
python main.py
```

Or run Streamlit directly:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Storage

The app creates a local SQLite database in `data/` on first run. It stores hazard reports, severity, GPS coordinates, timestamps, verification status, and reporter details.

## Limitations

- Route safety checks hazards along a straight line between source and destination, not real road paths.
- The "safest route" distance is an estimate, not a calculated alternative route.
- Verification status can be set by any user; there is no login or admin role yet.

## Future Work

- Real road routing for route safety
- User accounts and admin verification
- Accident prediction
- Public works and smart city integrations
- Push notifications and mobile deployment

## Author

SHEETHANA G.M
