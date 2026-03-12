# app/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    # =========================
    # INPUT
    # =========================
    # INPUT_PATH: str = "rtsp://admin:Qwerty12@192.168.199.90:554/Streaming/Channels/101"
    INPUT_PATH: str = "/home/muhammad/Waste_Monitoring_System/videos/video2.1.mp4"

    IMAGE_EXTS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    VIDEO_EXTS: tuple = (".mp4", ".avi", ".mkv", ".mov", ".m4v")

    # =========================
    # OUTPUT
    # =========================
    OUTPUT_DIR: Path = Path("/home/muhammad/serverga/output")
    EVENTS_DIR_NAME: str = "images"

    SHOW: bool = True             # server/headless uchun False
    SAVE_VIDEO: bool = True
    OUTPUT_VIDEO_NAME: str = "output_video2.1.mp4"
    OUTPUT_VIDEO_FPS: float = 20.0

    # =========================
    # MODELS
    # =========================
    DET_MODEL: str = "/home/muhammad/serverga/models/detection_best.pt"
    CLS_MODEL: str = "/home/muhammad/serverga/models/classification_best.pt"

    IMGSZ_DET: int = 640
    CONF_DET: float = 0.25
    IOU_DET: float = 0.45
    TRACKER: str = "bytetrack.yaml"
    PAD: float = 0.10
    IMGSZ_CLS: int = 224

    # =========================
    # STABILIZE
    # =========================
    WINDOW: int = 12
    FULL_TH: float = 0.75
    EMPTY_TH: float = 0.75

    # Hozir kamerada nechta konteyner ko'rinsa,
    # o'shalarning HAMMASI FULL bo'lsa alert
    MIN_BINS_VISIBLE: int = 1

    # =========================
    # ALL FULL EVENT
    # =========================
    ALLFULL_HOLD_SEC: float = 5.0          # birinchi marta yuborishdan oldin kutish
    ALLFULL_COOLDOWN_SEC: int = 180        # 3 minut to'liq bo'lsa qayta yuborish
    EVENT_IMAGE_MAX_W: int = 960
    EVENT_IMAGE_JPEG_QUALITY: int = 85
    SAVE_EVENT_FRAMES: bool = True

    # =========================
    # BACKEND
    # =========================
    SEND_TO_BACKEND: bool = True
    BACKEND_URL: str = "http://10.40.119.225:8082"
    CAMERA_ID: str = "CAM-NEW-002"
    BACKEND_TIMEOUT_SEC: float = 5.0
    JWT_TOKEN: str = ""

    # =========================
    # LIVE SOURCE / RTSP
    # =========================
    RECONNECT_DELAY_SEC: float = 1.0
