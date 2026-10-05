"""Small WoS REST client with conservative rate limiting and retry handling."""
from __future__ import annotations
import os, time, logging
from typing import Any
import requests

class WoSClient:
    def __init__(self, api_type: str = "auto", timeout: int = 45, max_retries: int = 6,
                 requests_per_second: float = 4.0, database: str = "WOS", on_request=None):
        self.key = os.environ.get("WOS_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("WOS_API_KEY environment variable is required.")
        self.api_type = "starter" if api_type == "auto" else api_type
        if self.api_type not in {"starter", "expanded"}:
            raise ValueError("api_type must be auto, starter, or expanded")
        self.url = ("https://api.clarivate.com/apis/wos-starter/v1/documents" if self.api_type == "starter"
                    else "https://api.clarivate.com/api/wos")
        self.timeout, self.max_retries = timeout, max_retries
        self.interval = 1.0 / min(requests_per_second, 5.0)
        self.database = database
        self.on_request = on_request
        self.session = requests.Session()
        self.session.headers.update({"X-ApiKey": self.key, "Accept": "application/json"})
        self.last_request = 0.0
        self.log = logging.getLogger("wos")

    def get(self, query: str, page: int = 1, limit: int = 50) -> dict[str, Any]:
        params = {"q": query, "page": page, "limit": min(limit, 50), "sortField": "RS+D"}
        if self.api_type == "expanded":
            params["databaseId"] = self.database
        for attempt in range(self.max_retries + 1):
            wait = self.interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                if self.on_request:
                    self.on_request()
                response = self.session.get(self.url, params=params, timeout=self.timeout)
                self.last_request = time.monotonic()
                if response.status_code == 200:
                    data = response.json()
                    if self.api_type == "auto":
                        self.api_type = "starter" if "metadata" in data and "hits" in data else "expanded"
                    return data
                if response.status_code in (401, 403):
                    raise PermissionError(f"WoS authentication/access denied (HTTP {response.status_code}); response body suppressed.")
                retryable = response.status_code == 429 or 500 <= response.status_code < 600
                if not retryable or attempt == self.max_retries:
                    detail = response.text[:300].replace(self.key, "[REDACTED]").replace("\n", " ")
                    raise RuntimeError(f"WoS HTTP {response.status_code}; {detail}")
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(120.0, 2 ** attempt)
                self.log.warning("WoS HTTP %s; retry %s/%s after %.1fs", response.status_code, attempt + 1, self.max_retries, delay)
                time.sleep(delay)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"WoS request failed after retries ({type(exc).__name__}).") from None
                delay = min(120.0, 2 ** attempt)
                self.log.warning("WoS network error; retry %s/%s after %.1fs", attempt + 1, self.max_retries, delay)
                time.sleep(delay)
        raise RuntimeError("WoS request retries exhausted")
