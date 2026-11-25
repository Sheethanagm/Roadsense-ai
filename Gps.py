from __future__ import annotations

from functools import lru_cache

from geopy.geocoders import GoogleV3, Nominatim


DEFAULT_LATITUDE = 13.0827
DEFAULT_LONGITUDE = 80.2707


@lru_cache(maxsize=16)
def reverse_geocode(latitude: float, longitude: float) -> str:
    api_key = None
    try:
        from os import getenv

        api_key = getenv("GOOGLE_MAPS_API_KEY", "").strip() or None
    except Exception:
        api_key = None

    services = []
    if api_key:
        services.append(GoogleV3(api_key=api_key, timeout=10))
    services.append(Nominatim(user_agent="road-hazard-system", timeout=10))

    for geocoder in services:
        try:
            location = geocoder.reverse(f"{latitude}, {longitude}", exactly_one=True)
            if location and getattr(location, "address", ""):
                return location.address
        except Exception:
            continue
    return ""


def resolve_location(latitude: float, longitude: float) -> str:
    address = reverse_geocode(latitude, longitude)
    if address:
        return address
    return f"Latitude {latitude:.6f}, Longitude {longitude:.6f}"


@lru_cache(maxsize=32)
def geocode_location(query: str) -> tuple[float, float, str]:
    cleaned = query.strip()
    if not cleaned:
        return DEFAULT_LATITUDE, DEFAULT_LONGITUDE, ""

    api_key = None
    try:
        from os import getenv

        api_key = getenv("GOOGLE_MAPS_API_KEY", "").strip() or None
    except Exception:
        api_key = None

    services = []
    if api_key:
        services.append(GoogleV3(api_key=api_key, timeout=10))
    services.append(Nominatim(user_agent="road-hazard-system", timeout=10))

    for geocoder in services:
        try:
            location = geocoder.geocode(cleaned, exactly_one=True)
            if location:
                address = getattr(location, "address", cleaned)
                return float(location.latitude), float(location.longitude), address
        except Exception:
            continue
    return DEFAULT_LATITUDE, DEFAULT_LONGITUDE, ""
