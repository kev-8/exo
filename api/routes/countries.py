"""GET /api/countries — list of tracked countries with geo metadata."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

COUNTRY_META: dict[str, dict] = {
    "US": {"name": "United States", "flag": "🇺🇸", "lat":  37.09, "lon": -95.71, "region": "North America"},
    "RU": {"name": "Russia",        "flag": "🇷🇺", "lat":  61.52, "lon": 105.32, "region": "Eastern Europe"},
    "CN": {"name": "China",         "flag": "🇨🇳", "lat":  35.86, "lon": 104.20, "region": "East Asia"},
    "UA": {"name": "Ukraine",       "flag": "🇺🇦", "lat":  48.38, "lon":  31.17, "region": "Eastern Europe"},
    "IL": {"name": "Israel",        "flag": "🇮🇱", "lat":  31.05, "lon":  34.85, "region": "Middle East"},
    "IR": {"name": "Iran",          "flag": "🇮🇷", "lat":  32.43, "lon":  53.69, "region": "Middle East"},
    "IN": {"name": "India",         "flag": "🇮🇳", "lat":  20.59, "lon":  78.96, "region": "South Asia"},
    "PK": {"name": "Pakistan",      "flag": "🇵🇰", "lat":  30.38, "lon":  69.35, "region": "South Asia"},
    "KP": {"name": "North Korea",   "flag": "🇰🇵", "lat":  40.34, "lon": 127.51, "region": "East Asia"},
    "TW": {"name": "Taiwan",        "flag": "🇹🇼", "lat":  23.70, "lon": 120.96, "region": "East Asia"},
    "HT": {"name": "Haiti",         "flag": "🇭🇹", "lat":  18.97, "lon": -72.29, "region": "Caribbean"},
    "BR": {"name": "Brazil",        "flag": "🇧🇷", "lat": -14.24, "lon": -51.93, "region": "South America"},
    "MX": {"name": "Mexico",        "flag": "🇲🇽", "lat":  23.63, "lon": -102.55,"region": "North America"},
    "NG": {"name": "Nigeria",       "flag": "🇳🇬", "lat":   9.08, "lon":   8.68, "region": "West Africa"},
    "KE": {"name": "Kenya",         "flag": "🇰🇪", "lat":  -0.02, "lon":  37.91, "region": "East Africa"},
    "ZA": {"name": "South Africa",  "flag": "🇿🇦", "lat": -30.56, "lon":  22.94, "region": "Southern Africa"},
    "FR": {"name": "France",        "flag": "🇫🇷", "lat":  46.23, "lon":   2.21, "region": "Europe"},
    "GB": {"name": "United Kingdom","flag": "🇬🇧", "lat":  55.38, "lon":  -3.44, "region": "Europe"},
    "MY": {"name": "Malaysia",      "flag": "🇲🇾", "lat":   4.21, "lon": 108.07, "region": "Southeast Asia"},
    "CL": {"name": "Chile",         "flag": "🇨🇱", "lat": -35.68, "lon": -71.54, "region": "South America"},
    "KR": {"name": "South Korea",   "flag": "🇰🇷", "lat":  35.91, "lon": 127.77, "region": "East Asia"},
    "JP": {"name": "Japan",         "flag": "🇯🇵", "lat":  36.20, "lon": 138.25, "region": "East Asia"},
    "DE": {"name": "Germany",       "flag": "🇩🇪", "lat":  51.17, "lon":  10.45, "region": "Europe"},
    "NL": {"name": "Netherlands",   "flag": "🇳🇱", "lat":  52.13, "lon":   5.29, "region": "Europe"},
    "AE": {"name": "United Arab Emirates", "flag": "🇦🇪", "lat": 23.42, "lon": 53.85, "region": "Middle East"},
    "SG": {"name": "Singapore",     "flag": "🇸🇬", "lat":   1.35, "lon": 103.82, "region": "Southeast Asia"},
    "VN": {"name": "Vietnam",       "flag": "🇻🇳", "lat":  14.06, "lon": 108.28, "region": "Southeast Asia"},
    "ID": {"name": "Indonesia",     "flag": "🇮🇩", "lat":  -0.79, "lon": 113.92, "region": "Southeast Asia"},
    "CD": {"name": "DR Congo",      "flag": "🇨🇩", "lat":  -4.04, "lon":  21.76, "region": "Central Africa"},
    "GN": {"name": "Guinea",        "flag": "🇬🇳", "lat":   9.95, "lon":  -9.70, "region": "West Africa"},
}


class CountryResponse(BaseModel):
    iso2: str
    name: str
    flag: str
    lat: float
    lon: float
    region: str


@router.get("/countries", response_model=list[CountryResponse])
def list_countries():
    return [CountryResponse(iso2=iso2, **meta) for iso2, meta in COUNTRY_META.items()]
