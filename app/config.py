# app/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:

    # INPUT
    INPUT_PATH: str = "rtsp://admin:Qwerty12@185.203.237.57:10554/Streaming/Channels/101"  # rtsp yoki video fayl yo'li
    # INPUT_PATH: str = "/home/muhammad/serverga/videos/video7.mp4"

    IMAGE_EXTS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    VIDEO_EXTS: tuple = (".mp4", ".avi", ".mkv", ".mov", ".m4v")


    # OUTPUT

    OUTPUT_DIR: Path = Path("/home/muhammad/serverga/output")
    EVENTS_DIR_NAME: str = "images"

    SHOW: bool = True
    SAVE_VIDEO: bool = False
    OUTPUT_VIDEO_NAME: str = "output_video7.mp4"
    OUTPUT_VIDEO_FPS: float = 20.0


    # MODELS

    DET_MODEL: str = "/home/muhammad/serverga/models/detection_best.pt"
    CLS_MODEL: str = "/home/muhammad/serverga/models/classification_best.pt"

    IMGSZ_DET: int = 640
    CONF_DET: float = 0.25
    IOU_DET: float = 0.45
    TRACKER: str = "bytetrack.yaml"
    PAD: float = 0.10
    IMGSZ_CLS: int = 224


    # STABILIZE

    WINDOW: int = 12
    FULL_TH: float = 0.75
    EMPTY_TH: float = 0.75


    MIN_BINS_VISIBLE: int = 1

    # EVENT
    ALLFULL_HOLD_SEC: float = 5.0
    ALLFULL_COOLDOWN_SEC: int = 180
    EVENT_IMAGE_MAX_W: int = 960
    EVENT_IMAGE_JPEG_QUALITY: int = 85
    SAVE_EVENT_FRAMES: bool = True   # True = localga ham saqlaydi, False = faqat backendga yuboradi
    # BACKEND

    SEND_TO_BACKEND: bool = True
    BACKEND_URL: str = "http://185.203.237.55:8082"
    CAMERA_ID: str = "CAM-NEW-002"
    BACKEND_TIMEOUT_SEC: float = 5.0
    JWT_TOKEN: str = ""

    # LIVE
    RECONNECT_DELAY_SEC: float = 2.0
