# app/backend_client.py
from __future__ import annotations

from datetime import datetime, timezone
import requests


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class BackendClient:
    def __init__(
        self,
        base_url: str,
        camera_id: str,
        enabled: bool = True,
        timeout_sec: float = 5.0,
        jwt_token: str = "",
        endpoint: str = "/api/ai/trashbin-status",
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.endpoint = endpoint
        self.camera_id = camera_id
        self.enabled = bool(enabled)
        self.timeout_sec = float(timeout_sec)
        self.jwt_token = jwt_token or ""
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _build_headers(self) -> dict[str, str]:
        headers = {}
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        return headers

    def send_status(
        self,
        *,
        is_full: bool,
        status: str,
        confidence: float | None = None,
        det_conf: float | None = None,
        image_base64: str | None = None,
        cls_conf: float | None = None,
    ) -> bool:
        if confidence is None and cls_conf is not None:
            confidence = cls_conf

        if not self.enabled:
            return True

        if not self.base_url:
            print("[BACKEND] WARN base_url bo'sh")
            return False
        
        payload = {
            "camera_id": self.camera_id,
            "status": str(status).upper(),
            "is_full": bool(is_full),
            "ts": utc_iso_z(),
        }


        if confidence is not None:
            payload["confidence"] = float(confidence)
        if det_conf is not None:
            payload["det_conf"] = float(det_conf)
        if image_base64:
            payload["imageBase64"] = image_base64



        try:
            r = self.session.post(
                f"{self.base_url}{self.endpoint}",
                json=payload,
                headers=self._build_headers(),
                timeout=self.timeout_sec,
            )
            if 200 <= r.status_code < 300:
                print(f"[BACKEND] OK status={r.status_code}")
                return True

            print(f"[BACKEND] WARN status={r.status_code} resp={r.text[:300]}")
            return False
        except requests.RequestException as e:
            print(f"[BACKEND] WARN request failed: {e}")
            return False
        
