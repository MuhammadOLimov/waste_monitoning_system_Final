# app/utils.py
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
import cv2


def clamp(v: int, a: int, b: int) -> int:
    return max(a, min(b, v))


def expand_box(x1, y1, x2, y2, w, h, pad=0.10):
    bw = x2 - x1
    bh = y2 - y1

    x1 = x1 - bw * pad
    y1 = y1 - bh * pad
    x2 = x2 + bw * pad
    y2 = y2 + bh * pad

    x1 = clamp(int(x1), 0, w - 1)
    y1 = clamp(int(y1), 0, h - 1)
    x2 = clamp(int(x2), 0, w - 1)
    y2 = clamp(int(y2), 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def resize_max_width(frame_bgr, max_w: int):
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr

    scale = max_w / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_base64_jpg(frame_bgr, jpeg_quality: int = 85) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def draw_label(img, x1: int, y1: int, text: str, color):
    y = max(25, y1 - 10)
    cv2.putText(img, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def now_local_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_event_frame(frame_bgr, events_dir: Path, camera_id: str, prefix: str = "allfull") -> Path:
    events_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{camera_id}_{now_local_str()}.jpg"
    out_path = events_dir / filename
    cv2.imwrite(str(out_path), frame_bgr)
    return out_path