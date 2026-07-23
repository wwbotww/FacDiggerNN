"""Small EODHD HTTP client with secret-safe caching and a persistent call budget."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote


class EODHDError(RuntimeError):
    """Base error raised by the EODHD adapter."""


class EODHDBudgetError(EODHDError):
    """Raised before a request would exceed the configured daily budget."""


class ResponseLike(Protocol):
    status_code: int
    headers: dict[str, str]

    def json(self) -> Any: ...


class Transport(Protocol):
    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> ResponseLike: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DailyCallBudget:
    """Conservative single-process guard for free-plan request usage."""

    def __init__(self, path: Path, limit: int, now: Callable[[], datetime] = _utc_now) -> None:
        self.path = path
        self.limit = limit
        self.now = now

    def _read(self) -> dict[str, Any]:
        today = self.now().date().isoformat()
        if not self.path.is_file():
            return {"date_utc": today, "network_attempts": 0}
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"date_utc": today, "network_attempts": 0}
        if state.get("date_utc") != today:
            return {"date_utc": today, "network_attempts": 0}
        return state

    def reserve(self, cost: int = 1) -> int:
        if cost < 1:
            raise ValueError("EODHD API call cost must be positive")
        state = self._read()
        used = int(state.get("api_calls", state.get("network_attempts", 0)))
        if used + cost > self.limit:
            raise EODHDBudgetError(
                f"EODHD daily API-call budget exhausted ({used}+{cost}>{self.limit}, UTC day)"
            )
        state["network_attempts"] = int(state.get("network_attempts", 0)) + 1
        state["api_calls"] = used + cost
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return int(state["api_calls"])

    def status(self) -> dict[str, Any]:
        state = self._read()
        used = int(state.get("api_calls", state.get("network_attempts", 0)))
        return {**state, "limit": self.limit, "remaining": max(self.limit - used, 0)}


class EODHDClient:
    """JSON-only client whose cache identity never contains the API token."""

    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        cache_dir: Path,
        budget: DailyCallBudget,
        timeout_seconds: float = 30.0,
        cache_ttl_hours: int = 24,
        max_retries: int = 2,
        refresh: bool = False,
        transport: Transport | None = None,
        now: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_token:
            raise EODHDError("EODHD API token is empty")
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.cache_dir = cache_dir
        self.budget = budget
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_hours = cache_ttl_hours
        self.max_retries = max_retries
        self.refresh = refresh
        self.now = now
        self.sleep = sleep
        self.transport = transport or self._default_transport()
        self.request_log: list[dict[str, Any]] = []

    @staticmethod
    def _default_transport() -> Transport:
        try:
            import requests
        except ImportError as exc:
            raise EODHDError(
                "EODHD support requires the 'eodhd' extra: uv sync --extra eodhd"
            ) from exc
        session = requests.Session()
        session.headers.update({"User-Agent": "FacDiggerNN/0.1 EODHD adapter"})
        return session

    @staticmethod
    def _safe_path(path: str) -> str:
        return "/".join(quote(part, safe="-_.") for part in path.strip("/").split("/"))

    @staticmethod
    def _cache_identity(path: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": path.strip("/"),
            "params": {key: params[key] for key in sorted(params) if key != "api_token"},
        }

    def _cache_path(self, identity: dict[str, Any]) -> Path:
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return self.cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"

    def _read_cache(self, path: Path) -> tuple[Any, str] | None:
        if self.refresh or not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(envelope["fetched_at"])
        except (KeyError, ValueError, json.JSONDecodeError, OSError):
            return None
        if self.cache_ttl_hours == 0:
            return None
        if self.now() - fetched_at > timedelta(hours=self.cache_ttl_hours):
            return None
        return envelope["data"], envelope["fetched_at"]

    def _write_cache(self, path: Path, identity: dict[str, Any], data: Any) -> str:
        fetched_at = self.now().isoformat()
        envelope = {"fetched_at": fetched_at, "request": identity, "data": data}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        return fetched_at

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        call_cost: int = 1,
    ) -> Any:
        public_params = {**(params or {}), "fmt": "json"}
        identity = self._cache_identity(path, public_params)
        cache_path = self._cache_path(identity)
        cached = self._read_cache(cache_path)
        if cached is not None:
            data, fetched_at = cached
            self.request_log.append(
                {**identity, "cache_hit": True, "fetched_at": fetched_at, "status": 200}
            )
            return data

        request_params = {**public_params, "api_token": self.api_token}
        url = f"{self.base_url}/{self._safe_path(path)}"
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            self.budget.reserve(call_cost)
            try:
                response = self.transport.get(
                    url, params=request_params, timeout=self.timeout_seconds
                )
                last_status = int(response.status_code)
            except Exception as exc:
                if attempt < self.max_retries:
                    self.sleep(min(2**attempt, 4))
                    continue
                message = f"EODHD request failed for {identity['path']}: {type(exc).__name__}"
                raise EODHDError(message) from exc

            if last_status == 200:
                try:
                    data = response.json()
                except Exception as exc:
                    raise EODHDError(
                        f"EODHD returned non-JSON data for {identity['path']}"
                    ) from exc
                if isinstance(data, dict) and ("error" in data or "message" in data):
                    detail = data.get("message", data.get("error"))
                    raise EODHDError(f"EODHD API error for {identity['path']}: {detail}")
                fetched_at = self._write_cache(cache_path, identity, data)
                self.request_log.append(
                    {
                        **identity,
                        "cache_hit": False,
                        "fetched_at": fetched_at,
                        "status": 200,
                        "call_cost": call_cost,
                    }
                )
                return data

            retryable = last_status == 429 or last_status >= 500
            if retryable and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else min(2**attempt, 4)
                self.sleep(delay)
                continue
            raise EODHDError(
                f"EODHD request failed for {identity['path']} with HTTP {last_status}; "
                "check token permissions and subscription coverage"
            )
        raise EODHDError(f"EODHD request failed for {identity['path']} with HTTP {last_status}")
