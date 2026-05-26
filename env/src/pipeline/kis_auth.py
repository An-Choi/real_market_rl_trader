"""KIS Open API authentication (token issuance + disk caching)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests


PROD_URL = "https://openapi.koreainvestment.com:9443"
DEMO_URL = "https://openapivts.koreainvestment.com:29443"
TOKEN_ENDPOINT = "/oauth2/tokenP"


@dataclass
class KISConfig:
    env: str            # "demo" or "real"
    app_key: str
    app_secret: str
    token_cache_path: Path

    @classmethod
    def from_env(cls, token_cache_path: Path) -> "KISConfig":
        env = os.environ.get("KIS_ENV", "demo")
        app_key = os.environ.get("KIS_APP_KEY")
        app_secret = os.environ.get("KIS_APP_SECRET")
        if not app_key:
            raise RuntimeError("KIS_APP_KEY environment variable is not set")
        if not app_secret:
            raise RuntimeError("KIS_APP_SECRET environment variable is not set")
        if env not in ("demo", "real"):
            raise RuntimeError(f"KIS_ENV must be 'demo' or 'real', got {env!r}")
        return cls(
            env=env,
            app_key=app_key,
            app_secret=app_secret,
            token_cache_path=Path(token_cache_path),
        )


class KISAuth:
    """Issue and cache the KIS access token."""

    def __init__(self, config: KISConfig) -> None:
        self._config = config

    @property
    def base_url(self) -> str:
        return DEMO_URL if self._config.env == "demo" else PROD_URL

    @property
    def app_key(self) -> str:
        return self._config.app_key

    @property
    def app_secret(self) -> str:
        return self._config.app_secret

    def get_token(self) -> str:
        cached = self._read_cache()
        if cached is not None:
            return cached
        return self._issue_and_cache()

    def _read_cache(self) -> str | None:
        path = self._config.token_cache_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        expires_at = datetime.fromisoformat(data["expires_at"])
        if expires_at <= datetime.now() + timedelta(minutes=5):
            return None
        return data["access_token"]

    def _issue_and_cache(self) -> str:
        response = requests.post(
            self.base_url + TOKEN_ENDPOINT,
            json={
                "grant_type": "client_credentials",
                "appkey": self._config.app_key,
                "appsecret": self._config.app_secret,
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
        access_token = body["access_token"]
        expires_in = int(body.get("expires_in", 86400))
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        self._write_cache(access_token, expires_at)
        return access_token

    def _write_cache(self, access_token: str, expires_at: datetime) -> None:
        path = self._config.token_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "access_token": access_token,
            "expires_at": expires_at.isoformat(),
        }))
