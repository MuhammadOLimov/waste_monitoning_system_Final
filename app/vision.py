# app/vision.py
from __future__ import annotations

import cv2
from ultralytics import YOLO

from .utils import expand_box, draw_label


class VisionEngine:
    def __init__(
        self,
        det_model_path: str,
        cls_model_path: str,
        imgsz_det: int,
        conf_det: float,
        iou_det: float,
        tracker: str,
        pad: float,
        imgsz_cls: int,
    ):
        self.det_model = YOLO(det_model_path)
        self.cls_model = YOLO(cls_model_path)

        self.imgsz_det = int(imgsz_det)
        self.conf_det = float(conf_det)
        self.iou_det = float(iou_det)
        self.tracker = tracker
        self.pad = float(pad)
        self.imgsz_cls = int(imgsz_cls)

    def cls_predict(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self.imgsz_cls, self.imgsz_cls),
            interpolation=cv2.INTER_AREA
        )

        res = self.cls_model.predict(rgb, verbose=False)
        r0 = res[0]

        probs = r0.probs
        idx = int(probs.top1)
        p = float(probs.top1conf)
        name = str(r0.names[idx]).lower()

        return name, p

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        outputs = []

        results = self.det_model.track(
            frame,
            persist=True,
            imgsz=self.imgsz_det,
            conf=self.conf_det,
            iou=self.iou_det,
            tracker=self.tracker,
            verbose=False,
        )

        if not results:
            return annotated, outputs

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return annotated, outputs

        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            det_conf = float(b.conf[0].item())

            tid = int(b.id[0].item()) if b.id is not None else -1
            if tid == -1:
                continue

            box = expand_box(x1, y1, x2, y2, w, h, pad=self.pad)
            if box is None:
                continue

            xx1, yy1, xx2, yy2 = box
            crop = frame[yy1:yy2, xx1:xx2]
            if crop.size == 0:
                continue

            cls_name, cls_p = self.cls_predict(crop)

            is_full = (cls_name == "full")
            status = "FULL" if is_full else "EMPTY"

            color = (0, 0, 255) if is_full else (0, 255, 0)
            cv2.rectangle(annotated, (xx1, yy1), (xx2, yy2), color, 2)
            draw_label(
                annotated,
                xx1,
                yy1,
                f"id:{tid} det:{det_conf:.2f} {status} cls:{cls_p:.2f}",
                color
            )

            outputs.append(
                {
                    "id": tid,
                    "status": status,
                    "is_full": is_full,
                    "cls_conf": float(cls_p),
                    "det_conf": float(det_conf),
                }
            )

        return annotated, outputs