"""
National Weather Service Alerts API client.

Weather data comes from api.weather.gov.

Human-readable locations such as "Baltimore, MD" are converted to
latitude/longitude with Nominatim only for geocoding. Weather content
itself comes only from the National Weather Service.

The client also accepts a location already written as:
    "39.2904,-76.6122"
"""

import logging
import re
import requests

logger = logging.getLogger("weather-client")

class WeatherClient:
    BASE_URL = "https://api.weather.gov"
    GEOCODING_URL = "https://nominatim.openstreetmap.org/search"
    def __init__(
        self,
        user_agent: str = "DatabricksWeatherIntelligenceApp/1.0",
        timeout: int = 20,
    ):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/geo+json",
        })

    def _get(
        self,
        path: str,
        params: dict | None = None,
    ) -> dict:
        """Send a GET request to the NWS API."""
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

    @staticmethod
    def _parse_lat_lon(
        location: str,
    ) -> tuple[float, float] | None:
        """Return coordinates when location is already 'lat,lon'."""
        match = re.fullmatch(
            r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*",
            location,
        )
        if not match:
            return None

        latitude = float(match.group(1))
        longitude = float(match.group(2))
        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        return latitude, longitude

    def geocode_location(
        self,
        location: str,
    ) -> tuple[float, float]:
        """Resolve a city/state string to latitude and longitude."""
        existing_coordinates = self._parse_lat_lon(location)

        if existing_coordinates:
            return existing_coordinates
        response = requests.get(
            self.GEOCODING_URL,
            params={
                "q": location,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "us",
            },
            headers={
                "User-Agent": "DatabricksWeatherIntelligenceApp/1.0"
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

    def get_active_alerts(
        self,
        latitude: float,
        longitude: float,
        limit: int = 50,
    ) -> list[dict]:
        """Fetch active NWS alerts for one latitude/longitude point."""
        data = self._get(
            "/alerts/active",
            params={
                "point": f"{latitude},{longitude}",
            },
        )
        features = data.get("features", [])
        return features[:limit]

    @staticmethod
    def _build_narrative(
        properties: dict,
    ) -> str:
        """
        Build the free-text body that will later be chunked and embedded.
        """
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

    @classmethod
    def _normalize_alert(
        cls,
        alert: dict,
        location: str,
    ) -> dict | None:
        """Convert one NWS GeoJSON alert into the weather document schema."""
        alert_id = alert.get("id")

        if not alert_id:
            return None
        properties = alert.get("properties") or {}
        narrative_text = cls._build_narrative(
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
            "severity": properties.get("severity"),
            "certainty": properties.get("certainty"),
            "urgency": properties.get("urgency"),
            "issued_at": properties.get("sent"),
            "effective_at": properties.get("effective"),
            "onset_at": properties.get("onset"),
            "expires_at": properties.get("expires"),
            "sender_name": properties.get("senderName"),
            "area_desc": properties.get("areaDesc"),
            "payload": alert,
        }

    def get_weather_documents(
        self,
        location: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Resolve a location, fetch its active NWS alerts, and normalize them.
        """
        latitude, longitude = self.geocode_location(
            location
        )

        alerts = self.get_active_alerts(
            latitude=latitude,
            longitude=longitude,
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
            "Retrieved %d active NWS alerts for %s",
            len(documents),
            location,
        )
        return documents

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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
