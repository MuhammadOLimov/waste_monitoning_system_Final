# app/runner.py
from __future__ import annotations
import time
from pathlib import Path
import cv2

from .utils import frame_to_base64_jpg, resize_max_width, save_event_frame
from .vision import VisionEngine
from .backend_client import BackendClient
from .config import Settings


def open_capture(src: str):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Capture ochilmadi: {src}")
    return cap


def create_video_writer(path: Path, fps: float, w: int, h: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter ochilmadi: {path}")
    return writer


def _stable_status(history: dict, tid: int, is_full: bool, s: Settings) -> str:
    history.setdefault(tid, []).append(1 if is_full else 0)
    history[tid] = history[tid][-s.WINDOW:]

    if len(history[tid]) < s.WINDOW:
        return "UNK"

    full_ratio = sum(history[tid]) / s.WINDOW
    if full_ratio >= s.FULL_TH:
        return "FULL"
    if (1.0 - full_ratio) >= s.EMPTY_TH:
        return "EMPTY"
    return "UNK"


def _best_conf(outputs: list[dict], visible_ids: list[int]) -> tuple[float, float]:
    ids = set(visible_ids)
    best_cls, best_det = 0.0, 0.0
    for o in outputs:
        if o["id"] in ids:
            best_cls = max(best_cls, float(o["cls_conf"]))
            best_det = max(best_det, float(o["det_conf"]))
    return best_cls, best_det


def _send_event(frame, outputs, visible_ids, visible_count, full_count, backend, s) -> bool:
    event_frame = resize_max_width(frame, s.EVENT_IMAGE_MAX_W)

    if s.SAVE_EVENT_FRAMES:
        saved_path = save_event_frame(
            event_frame,
            events_dir=s.OUTPUT_DIR / s.EVENTS_DIR_NAME,
            camera_id=s.CAMERA_ID,
            prefix="allfull",
        )
        print(f"📸 Event frame saqlandi: {saved_path}")

    best_cls, best_det = _best_conf(outputs, visible_ids)
    img_b64 = frame_to_base64_jpg(event_frame, jpeg_quality=s.EVENT_IMAGE_JPEG_QUALITY)

    print(f"🔥 ALL VISIBLE BINS FULL CONFIRMED (visible={visible_count}, full={full_count}) -> SEND POST")

    return backend.send_status(
        is_full=True,
        status="FULL",
        cls_conf=best_cls,
        det_conf=best_det,
        image_base64=img_b64,
    )


def run_realtime(engine: VisionEngine, backend: BackendClient, s: Settings, src: str, is_live_source: bool):
    s.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_capture(src)
    writer = None
    out_video = s.OUTPUT_DIR / s.OUTPUT_VIDEO_NAME

    history = {}
    allfull_start = None
    next_alert_at = None
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if is_live_source:
                print("WARN: live frame kelmadi. reconnect...")
                cap.release()
                time.sleep(s.RECONNECT_DELAY_SEC)
                cap = open_capture(src)
                continue
            else:
                print("INFO: video tugadi")
                break

        annotated, outputs = engine.process_frame(frame)

        now = time.time()
        fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
        prev_t = now

        visible_ids, full_ids, empty_ids, unk_ids = [], [], [], []

        for o in outputs:
            tid = o["id"]
            visible_ids.append(tid)
            st = _stable_status(history, tid, o["is_full"], s)

            if st == "FULL":
                full_ids.append(tid)
            elif st == "EMPTY":
                empty_ids.append(tid)
            else:
                unk_ids.append(tid)

        visible_ids = sorted(set(visible_ids))
        full_ids = sorted(set(full_ids))
        empty_ids = sorted(set(empty_ids))
        unk_ids = sorted(set(unk_ids))

        visible_count = len(visible_ids)
        full_count = len(full_ids)
        empty_count = len(empty_ids)
        unk_count = len(unk_ids)

        all_full_now = visible_count >= s.MIN_BINS_VISIBLE and full_count == visible_count

        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"VISIBLE: {visible_count}  FULL: {full_count}  EMPTY: {empty_count}  UNK: {unk_count}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )
        cv2.putText(
            annotated,
            f"ALL_VISIBLE_FULL: {all_full_now}",
            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )

        if s.SAVE_VIDEO and writer is None:
            h, w = annotated.shape[:2]
            writer = create_video_writer(out_video, s.OUTPUT_VIDEO_FPS, w, h)
            print(f"✅ Video yozish boshlandi: {out_video}")

        if all_full_now:
            if allfull_start is None:
                allfull_start = now
                next_alert_at = None

            held = now - allfull_start
            cv2.putText(
                annotated,
                f"ALERT HOLD: {held:.1f}/{s.ALLFULL_HOLD_SEC:.1f}s",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

            if held >= s.ALLFULL_HOLD_SEC:
                if next_alert_at is None or now >= next_alert_at:
                    next_alert_at = now + s.ALLFULL_COOLDOWN_SEC
                    ok_sent = _send_event(frame, outputs, visible_ids, visible_count, full_count, backend, s)
                    print("✅ Backendga FULL holat yuborildi" if ok_sent else "⚠️ Backend qabul qilmadi")
        else:
            allfull_start = None
            next_alert_at = None

        if writer is not None:
            writer.write(annotated)

        if s.SHOW:
            cv2.imshow("Pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"✅ Video saqlandi: {out_video}")
    if s.SHOW:
        cv2.destroyAllWindows()


def run_image(engine: VisionEngine, s: Settings, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Rasm ochilmadi: {img_path}")

    annotated, _ = engine.process_frame(img)
    s.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = s.OUTPUT_DIR / f"{img_path.stem}_pipeline{img_path.suffix}"
    cv2.imwrite(str(out_path), annotated)
    print(f"✅ Image saved: {out_path}")

    if s.SHOW:
        cv2.imshow("Image Pipeline", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_app(s: Settings):
    engine = VisionEngine(
        det_model_path=s.DET_MODEL,
        cls_model_path=s.CLS_MODEL,
        imgsz_det=s.IMGSZ_DET,
        conf_det=s.CONF_DET,
        iou_det=s.IOU_DET,
        tracker=s.TRACKER,
        pad=s.PAD,
        imgsz_cls=s.IMGSZ_CLS,
    )

    backend = BackendClient(
        base_url=s.BACKEND_URL,
        camera_id=s.CAMERA_ID,
        enabled=s.SEND_TO_BACKEND,
        timeout_sec=s.BACKEND_TIMEOUT_SEC,
        jwt_token=s.JWT_TOKEN,
    )

    p = Path(s.INPUT_PATH)
    if p.exists():
        ext = p.suffix.lower()
        if ext in s.IMAGE_EXTS:
            run_image(engine, s, p)
            return
        if ext in s.VIDEO_EXTS:
            run_realtime(engine, backend, s, str(p), is_live_source=False)
            return
        raise ValueError(f"Noma'lum fayl turi: {ext}")

    run_realtime(engine, backend, s, s.INPUT_PATH, is_live_source=True)
