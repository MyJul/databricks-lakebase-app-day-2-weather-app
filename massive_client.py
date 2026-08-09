"""
National Weather Service API client.
The NWS API is the sole weather data source.
Responsibilities:
    - Resolve a human-readable location to latitude/longitude
    - Resolve latitude/longitude to an NWS forecast grid
    - Retrieve active alerts for the location
    - Normalize NWS alerts into weather_news document records
NWS API:
    https://api.weather.gov
NWS requires a User-Agent identifying the application.
"""

import hashlib
import logging
from datetime import datetime, timezone
import requests

logger = logging.getLogger("weather-client")

class WeatherClient:
   
    BASE_URL = "https://api.weather.gov"
    GEOCODING_URL = "https://nominatim.openstreetmap.org/search"

    def __init__(
        self,
        user_agent: str = "DatabricksWeatherApp/1.0",
        timeout: int = 20,
    ):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/geo+json",
        })

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = (
            path
            if path.startswith("http")
            else f"{self.BASE_URL}{path}"
        )

        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    def geocode_location(self, location: str) -> tuple[float, float]:
       
        response = requests.get(
            self.GEOCODING_URL,
            params={
                "q": location,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "us",
            },
            headers={
                "User-Agent": "DatabricksWeatherApp/1.0"
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        results = response.json()

        if not results:
            raise ValueError(
                f"Unable to resolve location: {location}"
            )

        return (
            float(results[0]["lat"]),
            float(results[0]["lon"]),
        )

    def get_point_metadata(
        self,
        latitude: float,
        longitude: float,
    ) -> dict:
        return self._get(
            f"/points/{latitude},{longitude}"
        )

    def get_active_alerts(
        self,
        latitude: float,
        longitude: float,
        limit: int = 50,
    ) -> list[dict]:
        data = self._get(
            "/alerts/active",
            params={
                "point": f"{latitude},{longitude}",
            },
        )

        features = data.get("features", [])
        return features[:limit]

    def get_forecast(
        self,
        point_metadata: dict,
    ) -> dict:
        properties = point_metadata.get("properties", {})
        forecast_url = properties.get("forecast")
        if not forecast_url:
            raise ValueError(
                "NWS /points response did not contain a forecast URL"
            )
        return self._get(forecast_url)

    @staticmethod
    def _build_narrative(properties: dict) -> str:
        parts = []
        event = properties.get("event")
        headline = properties.get("headline")
        description = properties.get("description")
        instruction = properties.get("instruction")
        if event:
            parts.append(f"Event: {event}")
        if headline:
            parts.append(f"Headline: {headline}")
        if description:
            parts.append(
                f"Description:\n{description}"
            )
        if instruction:
            parts.append(
                f"Instructions:\n{instruction}"
            )
        return "\n\n".join(parts).strip()

    @staticmethod
    def _normalize_alert(
        alert: dict,
        location: str,
    ) -> dict | None:
        alert_id = alert.get("id")
        if not alert_id:
            return None
        properties = alert.get("properties", {})
        narrative_text = WeatherClient._build_narrative(
            properties
        )

        if not narrative_text:
            return None
        return {
            "id": str(alert_id),
            "location": location,
            "source_type": "alert",
            "event": properties.get("event"),
            "headline": properties.get("headline"),
            "narrative_text": narrative_text,
            "description": properties.get("description"),
            "instruction": properties.get("instruction"),
            "severity": properties.get("severity"),
            "certainty": properties.get("certainty"),
            "urgency": properties.get("urgency"),
            "effective_at": properties.get("effective"),
            "onset_at": properties.get("onset"),
            "expires_at": properties.get("expires"),
            "sender_name": properties.get("senderName"),
            "sender_id": properties.get("sender"),
            "area_desc": properties.get("areaDesc"),
            "geocode": properties.get("geocode"),
            "geometry": alert.get("geometry"),
            "payload": alert,
        }

    def get_weather_documents(
        self,
        location: str,
        limit: int = 50,
    ) -> list[dict]:
        latitude, longitude = self.geocode_location(
            location
        )

        self.get_point_metadata(
            latitude,
            longitude,
        )

        alerts = self.get_active_alerts(
            latitude,
            longitude,
            limit=limit,
        )

        documents = []
        for alert in alerts:
            document = self._normalize_alert(
                alert,
                location,
            )
            if document:
                documents.append(document)
        logger.info(
            "Retrieved %d weather alerts for %s",
            len(documents),
            location,
        )
        return documents

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )
    client = WeatherClient()

    documents = client.get_weather_documents(
        "Baltimore, MD",
        limit=10,
    )
    for document in documents:
        print(
            document["id"],
            document["event"],
            document["headline"],
        )

