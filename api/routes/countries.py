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
