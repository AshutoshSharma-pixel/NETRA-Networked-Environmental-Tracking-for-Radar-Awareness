"""
vision/radar_cv.py

Professional perception-grade radar visualization.
- Uses Ultralytics YOLOv8 for detection
- Centroid-based multi-object tracking (persistent IDs)
- Exponential smoothing of positions to reduce jitter
- Velocity computation, predicted ghost positions
- Threat scoring (confidence, speed, stability) and color coding
- Fading trails, uncertainty halos, and fading when tracks are lost

Run: python vision/radar_cv.py

Perception and visualization only.
"""

import math
import time
from typing import List, Tuple
from collections import deque

try:
    import cv2
    import numpy as np
    import pygame
    from ultralytics import YOLO
except Exception as e:
    print("Required packages not available. Install: pip install ultralytics opencv-python numpy pygame")
    raise

# --- Configuration ---------------------------------------------------------
SCREEN_SIZE = (900, 700)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
MODEL_CONF_PRE = 0.06  # low conf used to get candidates
DETECTION_CONF_BASE = 0.35
DETECTION_CONF_MIN = 0.08
H_FOV_DEGREES = 60.0
MAX_TRACK_MISSED = 12
SMOOTH_ALPHA = 0.6
PREDICT_SEC = 1.0
TRAIL_LEN = 24
FPS = 30

# Colors
BG_COLOR = (10, 12, 15)
RADAR_GREEN = (0, 220, 100)
DARK_GREEN = (6, 30, 12)

# COCO names
COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# ---------------------------------------------------------------------------
# Utility: map bounding box to radar polar coordinates
# ---------------------------------------------------------------------------

def map_detection_to_radar(bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[float, float]:
    """Return (angle_rad, distance_norm) for a bbox in image pixels.

    angle: -FOV/2 .. +FOV/2 based on bbox center x
    distance_norm: 0 (close) .. 1 (far) derived from bbox area fraction
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    nx = (cx / frame_w) - 0.5
    angle = nx * math.radians(H_FOV_DEGREES)

    area_frac = (w * h) / float(frame_w * frame_h)
    A0 = 0.25
    Amin = 0.0008
    if area_frac >= A0:
        dist_norm = 0.0
    else:
        dist_norm = (A0 - area_frac) / max(1e-6, (A0 - Amin))
        dist_norm = max(0.0, min(1.0, dist_norm))
    return angle, dist_norm


# ---------------------------------------------------------------------------
# Detection: preprocess and run YOLOv8
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Perform lighting normalization using LAB + CLAHE on the L channel.

    The returned frame is intended for detection only (model input).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to the L channel to normalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    # Merge and convert back to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    proc = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    # Mild smoothing to reduce small artifacts
    proc = cv2.GaussianBlur(proc, (3, 3), 0)
    return proc


def detect_objects(model: YOLO, frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
    H, W = frame.shape[:2]
    proc = preprocess_frame(frame)
    img_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, imgsz=640, conf=MODEL_CONF_PRE, verbose=False)
    if not results:
        return []
    res = results[0]

    # extract arrays
    try:
        boxes = np.array(res.boxes.xyxy.cpu()) if hasattr(res.boxes.xyxy, 'cpu') else np.array(res.boxes.xyxy)
        confs = np.array(res.boxes.conf.cpu()) if hasattr(res.boxes.conf, 'cpu') else np.array(res.boxes.conf)
        clss = np.array(res.boxes.cls.cpu()) if hasattr(res.boxes.cls, 'cpu') else np.array(res.boxes.cls)
    except Exception:
        # fallback
        boxes, confs, clss = [], [], []
        for b in getattr(res, 'boxes', []):
            xy = b.xyxy[0] if hasattr(b.xyxy, '__len__') else b.xyxy
            coords = xy.cpu().numpy() if hasattr(xy, 'cpu') else np.array(xy)
            boxes.append(coords)
            confs.append(float(b.conf[0]) if hasattr(b.conf, '__len__') else float(b.conf))
            clss.append(int(b.cls[0]) if hasattr(b.cls, '__len__') else int(b.cls))
        if boxes:
            boxes = np.vstack(boxes)
            confs = np.array(confs)
            clss = np.array(clss)
        else:
            return []

    detections = []
    for i in range(boxes.shape[0]):
        x1f, y1f, x2f, y2f = boxes[i]
        conf = float(confs[i]) if confs is not None else 0.0
        cls_id = int(clss[i]) if clss is not None else -1
        x1 = int(max(0, min(W - 1, x1f)))
        y1 = int(max(0, min(H - 1, y1f)))
        x2 = int(max(0, min(W - 1, x2f)))
        y2 = int(max(0, min(H - 1, y2f)))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        size_frac = math.sqrt((bw * bh) / float(W * H))
        dyn_thresh = max(DETECTION_CONF_MIN, DETECTION_CONF_BASE * (0.5 + size_frac))
        if conf < dyn_thresh:
            continue
        label = COCO_NAMES[cls_id] if 0 <= cls_id < len(COCO_NAMES) else f"cls{cls_id}"
        detections.append((label, (x1, y1, bw, bh), conf))
    return detections


# ---------------------------------------------------------------------------
# Tracking: centroid matching, smoothing, velocity, prediction, threat
# ---------------------------------------------------------------------------


class Track:
    def __init__(self, tid: int, label: str, bbox: Tuple[int, int, int, int], conf: float, tstamp: float):
        self.id = tid
        self.label = label
        self.bbox = bbox
        x, y, w, h = bbox
        self.cx = x + w / 2.0
        self.cy = y + h / 2.0
        self.vx = 0.0
        self.vy = 0.0
        self.conf = conf
        self.last_seen = tstamp
        self.age = 1
        self.hits = 1
        self.missed = 0
        self.trail = deque(maxlen=TRAIL_LEN)
        self.trail.append((self.cx, self.cy))
        # classification memory
        self.class_history = deque(maxlen=12)
        self.conf_history = deque(maxlen=12)
        # initialize with first detection
        if label is not None:
            self.class_history.append(label)
            self.conf_history.append(conf)
        # stability and lifecycle
        self.quality = 'MED'  # LOW / MED / HIGH
        self.state = 'NEW'  # NEW / STABLE / FADING
        self.label_locked = False
        self._last_consensus = label
        self._consec_new = 0
        self.uncertain = False
        self.stability_score = self.stability()
        # sound/pulse related fields
        self.created_at = tstamp
        self.last_beep = 0.0
        self.last_tick = 0.0
        self.last_warning = 0.0
        self.pulse_until = 0.0
        # altitude estimation (normalized 0..1)
        self.altitude_norm = 0.5

    def update(self, bbox: Tuple[int, int, int, int], conf: float, tstamp: float, alpha: float = SMOOTH_ALPHA, detected_label: str = None, detected_conf: float = None, frame_wh: Tuple[int, int] = None):
        x, y, w, h = bbox
        new_cx = x + w / 2.0
        new_cy = y + h / 2.0
        dt = max(1e-4, tstamp - self.last_seen)
        a = max(0.12, min(0.95, alpha * (0.5 + conf)))
        prev_cx, prev_cy = self.cx, self.cy
        self.cx = a * new_cx + (1 - a) * self.cx
        self.cy = a * new_cy + (1 - a) * self.cy
        # velocity
        self.vx = (self.cx - prev_cx) / dt
        self.vy = (self.cy - prev_cy) / dt
        self.bbox = bbox
        self.conf = conf
        self.last_seen = tstamp
        self.age += 1
        self.hits += 1
        self.missed = 0
        self.trail.append((self.cx, self.cy))

        # altitude estimation (normalized 0..1) based on bbox area and vertical position
        # Smaller bbox area typically indicates higher altitude; vertical position biases estimate too
        if frame_wh is not None:
            fw, fh = frame_wh
            area = max(1.0, w * h)
            area_norm = min(1.0, area / float(max(1, fw * fh)))
            y_center = (y + h / 2.0) / max(1.0, fh)
            # altitude_norm: smaller area -> higher altitude; objects near top considered higher
            self.altitude_norm = float(max(0.0, min(1.0, (1.0 - area_norm) * 0.75 + y_center * 0.25)))
        else:
            # fallback if frame size unavailable
            self.altitude_norm = getattr(self, 'altitude_norm', 0.5)

        # classification memory update
        if detected_label is not None:
            # only accept if detection confidence above a minimal floor
            self.class_history.append(detected_label)
            self.conf_history.append(detected_conf if detected_conf is not None else conf)
            self._perform_label_decision(bbox)

        # update quality/state
        self._update_quality_and_state()

    def mark_missed(self):
        self.missed += 1
        self.age += 1
        self._update_quality_and_state()

    def predict(self, seconds: float) -> Tuple[float, float]:
        # smoothed linear prediction
        return (self.cx + self.vx * seconds, self.cy + self.vy * seconds)

    def stability(self) -> float:
        return max(0.0, min(1.0, float(self.hits) / float(max(1, self.age))))

    def threat_score(self, frame_wh: Tuple[int, int]) -> float:
        W, H = frame_wh
        diag = math.hypot(W, H)
        speed = math.hypot(self.vx, self.vy)
        speed_norm = max(0.0, min(1.0, speed / max(1.0, diag * 0.9)))
        conf_norm = max(0.0, min(1.0, self.conf))
        stab = self.stability()
        score = 0.4 * conf_norm + 0.45 * speed_norm + 0.15 * stab
        return max(0.0, min(1.0, score))

    def _update_quality_and_state(self):
        # quality from combined confidence and stability
        stab = self.stability()
        score = (self.conf * 0.6) + (stab * 0.4)
        if score >= 0.75:
            self.quality = 'HIGH'
        elif score >= 0.4:
            self.quality = 'MED'
        else:
            self.quality = 'LOW'
        # lifecycle state
        if self.hits <= 2 and self.missed == 0:
            self.state = 'NEW'
        elif self.missed > 0 and self.missed < 6:
            self.state = 'FADING'
        else:
            self.state = 'STABLE'
        self.stability_score = stab

    def _perform_label_decision(self, bbox: Tuple[int, int, int, int]):
        # Weighted majority voting
        weights = {}
        for lbl, cf in zip(self.class_history, self.conf_history):
            weights[lbl] = weights.get(lbl, 0.0) + max(0.01, cf)
        if not weights:
            return
        # determine top two
        items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_label, top_w = items[0]
        second_label, second_w = (items[1] if len(items) > 1 else (None, 0.0))
        total_w = sum(weights.values())

        # confidence gating
        CONF_CHANGE_THRESH = 0.45
        # require the most recent detection to have reasonable confidence to consider change
        recent_conf = self.conf_history[-1] if len(self.conf_history) > 0 else self.conf
        if recent_conf < CONF_CHANGE_THRESH and top_label != self.label:
            # ignore noisy change
            self.uncertain = False
            return

        # class-priority heuristics for ambiguous small objects
        # if ambiguous between phone and skateboard, use bbox aspect and area
        if second_label is not None and {top_label, second_label} >= {"cell phone", "skateboard"}:
            x, y, w, h = bbox
            area = w * h
            aspect = h / max(1.0, w)
            if aspect > 1.6 and area < (FRAME_WIDTH * FRAME_HEIGHT * 0.015):
                top_label = "cell phone"
            else:
                top_label = "skateboard"

        # decide uncertain state when top is not dominant
        dominance = top_w / max(1e-6, total_w)
        if second_label and (second_w / max(1e-6, total_w)) > 0.4:
            # ambiguous
            self.uncertain = True
            self.label = f"Uncertain ({top_label}/{second_label})"
        else:
            self.uncertain = False
            # stable change only after M consecutive consensus
            M = 3
            if top_label == self._last_consensus:
                self._consec_new += 1
            else:
                self._consec_new = 1
                self._last_consensus = top_label
            if self._consec_new >= M:
                # commit label change
                if top_label != self.label:
                    self.label = top_label
                # lock label if track is stable
                if self.stability() > 0.6 and self.hits > 4:
                    self.label_locked = True


class Tracker:
    def __init__(self, max_missed: int = MAX_TRACK_MISSED):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_missed = max_missed

    def update(self, detections: List[Tuple[str, Tuple[int, int, int, int], float]], tstamp: float, frame_size: Tuple[int, int]) -> List[Track]:
        W, H = frame_size
        assigned = set()
        det_centers = [((d[1][0] + d[1][2] / 2.0), (d[1][1] + d[1][3] / 2.0)) for d in detections]
        diag = math.hypot(W, H)
        dist_thresh = max(40.0, diag * 0.06)

        # match tracks greedily by nearest centroid
        for tr in self.tracks:
            best_d = None
            best_di = -1
            for di, d in enumerate(detections):
                if di in assigned:
                    continue
                dcx, dcy = det_centers[di]
                dval = math.hypot(dcx - tr.cx, dcy - tr.cy)
                if best_d is None or dval < best_d:
                    best_d = dval
                    best_di = di
            if best_d is not None and best_d <= dist_thresh:
                label, bbox, conf = detections[best_di]
                # pass label/conf and frame size to track update for temporal voting and altitude
                tr.update(bbox, conf, tstamp, alpha=SMOOTH_ALPHA, detected_label=label, detected_conf=conf, frame_wh=(W, H))
                assigned.add(best_di)
            else:
                tr.mark_missed()

        # create new tracks
        for di, det in enumerate(detections):
            if di in assigned:
                continue
            label, bbox, conf = det
            t = Track(self.next_id, label, bbox, conf, tstamp)
            # initialize altitude_norm for new track
            x, y, w, h = bbox
            area = max(1.0, w * h)
            area_norm = min(1.0, area / float(max(1, W * H)))
            y_center = (y + h / 2.0) / max(1.0, H)
            t.altitude_norm = float(max(0.0, min(1.0, (1.0 - area_norm) * 0.75 + y_center * 0.25)))
            self.next_id += 1
            self.tracks.append(t)

        # Prune stale tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks


# ---------------------------------------------------------------------------
# Visualization: radar UI and rendering helpers
# ---------------------------------------------------------------------------


def threat_color(score: float) -> Tuple[int, int, int]:
    s = max(0.0, min(1.0, score))
    if s < 0.5:
        # green -> yellow
        t = s / 0.5
        r = int(80 + (200 - 80) * t)
        g = 220
        b = int(60 - 20 * t)
    else:
        t = (s - 0.5) / 0.5
        r = int(200 + (55) * t)
        g = int(220 - (200 * t))
        b = 40
    return (min(255, r), max(0, g), b)


class RadarDisplay:
    """Patriot-style radar UI focused on disciplined aesthetics and readability.

    Visual features added:
    - Thicker outer ring with subtle inner shadow
    - Concentric rings with numeric distance labels (10,20,...)
    - Arc tick marks along outer ring
    - Reduced sweep ray count for smooth beam with bloom and softer tail
    - Pulsing center with tiny crosshair notches
    - Diamond target markers with soft outer halo and orientation notch
    - Labels with outline for contrast
    - Trails fading faster near center and slower near edges, with mild smoothing
    - HUD panel inside the scope showing detections, tracks, FPS
    - Subtle radial vignette and breathing glow

    All elements are constrained to remain inside the radar circle.
    """

    def __init__(self, size: Tuple[int, int]):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Perception Radar Console")
        self.clock = pygame.time.Clock()
        self.center = (size[0] // 2, size[1] // 2)
        self.radius = min(size) // 2 - 48
        self.sweep_angle = 0.0
        self.sweep_speed = math.radians(34)  # slightly slower
        # sweep velocity used for eased motion
        self.sweep_vel = self.sweep_speed
        self.sweep_smooth_k = 6.0
        self.sweep_blur_steps = 3
        self.sweep_blur_decay = 0.55
        self.font = pygame.font.SysFont('consolas', 14)
        self.small_font = pygame.font.SysFont('consolas', 12)
        self.hud_font = pygame.font.SysFont('consolas', 13)
        self.last_track_count = 0
        # colors constrained to palette
        self.green_dark = (8, 18, 10)
        self.green_mid = (12, 90, 40)
        # slightly desaturate for professional look
        self.green_light = tuple(max(0, int(c * 0.92)) for c in RADAR_GREEN)
        # store last dt for animations
        self._last_dt = 1.0 / FPS
        
    def update(self, dt: float):
        # clamp dt for stability
        dt = max(0.0, min(0.1, dt))
        self._last_dt = dt
        # eased sweep: softly adjust sweep velocity to target speed to create micro acceleration
        target = self.sweep_speed
        # small easing toward target velocity
        self.sweep_vel += (target - self.sweep_vel) * (1.0 - math.exp(-self.sweep_smooth_k * dt))
        # advance angle using eased velocity
        self.sweep_angle = (self.sweep_angle + self.sweep_vel * dt) % (2 * math.pi)

    def clear(self):
        # base fill
        self.screen.fill(BG_COLOR)
        # subtle breathing vignette over entire radar area
        cx, cy = self.center
        r = self.radius
        t = time.time()
        breathe = 0.95 + 0.05 * math.sin(t * 0.6)
        vignette = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        for i in range(r, 0, -8):
            alpha = int(10 * (1.0 - (i / r)) * breathe)
            pygame.draw.circle(vignette, (0, 10, 6, alpha), (r, r), i)
        self.screen.blit(vignette, (cx - r, cy - r), special_flags=pygame.BLEND_PREMULTIPLIED)

    def draw_scope(self):
        cx, cy = self.center
        r = self.radius

        # Outer ring with inner shadow
        outer = pygame.Surface((r * 2 + 16, r * 2 + 16), pygame.SRCALPHA)
        pygame.draw.circle(outer, (18, 80, 40, 220), (r + 8, r + 8), r + 8)
        pygame.draw.circle(outer, (0, 0, 0, 180), (r + 8, r + 8), r - 6)
        self.screen.blit(outer, (cx - r - 8, cy - r - 8), special_flags=pygame.BLEND_PREMULTIPLIED)

        # Inner dark fill
        pygame.draw.circle(self.screen, self.green_dark, (cx, cy), r - 12)

        # concentric rings and numeric labels
        rings = 6
        max_label = 60
        for i in range(1, rings):
            ri = int((r - 20) * i / rings)
            col = self.green_mid if i % 2 == 0 else (10, 70, 35)
            pygame.draw.circle(self.screen, col, (cx, cy), ri, 1)
            # numeric label at top of ring
            val = int(max_label * (i / rings))
            lbl = self.small_font.render(str(val), True, self.green_light)
            lx = cx - lbl.get_width() // 2
            ly = cy - ri - lbl.get_height() // 2
            # ensure inside circle
            if (lx - cx) ** 2 + (ly - cy) ** 2 <= (r - 24) ** 2:
                # slight shadow outline
                self.screen.blit(self._outlined(lbl), (lx, ly))

        # arc tick marks along outer ring
        tick_count = 36
        for i in range(tick_count):
            ang = (2 * math.pi) * (i / tick_count)
            out_x = cx + math.cos(ang) * (r - 6)
            out_y = cy + math.sin(ang) * (r - 6)
            in_x = cx + math.cos(ang) * (r - 14)
            in_y = cy + math.sin(ang) * (r - 14)
            pygame.draw.line(self.screen, (12, 60, 30), (in_x, in_y), (out_x, out_y), 1)

        # crosshair notches and bold axes
        pygame.draw.line(self.screen, self.green_light, (cx - r + 4, cy), (cx + r - 4, cy), 2)
        pygame.draw.line(self.screen, self.green_light, (cx, cy - r + 4), (cx, cy + r - 4), 2)
        # center small crosshair notches
        notch_len = 8
        pygame.draw.line(self.screen, self.green_light, (cx - notch_len, cy), (cx - 3, cy), 1)
        pygame.draw.line(self.screen, self.green_light, (cx + 3, cy), (cx + notch_len, cy), 1)
        pygame.draw.line(self.screen, self.green_light, (cx, cy - notch_len), (cx, cy - 3), 1)
        pygame.draw.line(self.screen, self.green_light, (cx, cy + 3), (cx, cy + notch_len), 1)

        # center pulsing glow
        t = time.time()
        pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 1.2))
        cr = int(6 * pulse)
        center_s = pygame.Surface((cr * 4, cr * 4), pygame.SRCALPHA)
        pygame.draw.circle(center_s, (self.green_light[0], self.green_light[1], self.green_light[2], int(120 * pulse)), (cr * 2, cr * 2), cr * 2)
        pygame.draw.circle(center_s, (self.green_light[0], self.green_light[1], self.green_light[2], 255), (cr * 2, cr * 2), 3)
        self.screen.blit(center_s, (cx - cr * 2, cy - cr * 2), special_flags=pygame.BLEND_PREMULTIPLIED)

        # HUD panel (top-left inside scope)
        hud_w = int(r * 0.6)
        hud_h = 56
        hud_x = cx - int(r * 0.9) + 10
        hud_y = cy - int(r * 0.85) + 10
        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        pygame.draw.rect(hud_surf, (4, 30, 14, 180), (0, 0, hud_w, hud_h), border_radius=6)
        # text lines: title and counts are drawn elsewhere to maintain outline style
        self.screen.blit(hud_surf, (hud_x, hud_y), special_flags=pygame.BLEND_PREMULTIPLIED)

        # After drawing HUD panel, draw performance meter and sensor status
        # sensor status: simple 'OK' if frame rate healthy
        fps = int(self.clock.get_fps())
        status = 'OK' if fps >= FPS * 0.6 else 'DEGRADED'
        status_s = self.small_font.render(f'SENSOR: {status}', True, self.green_light)
        self.screen.blit(self._outlined(status_s), (cx - r + 16, cy - r + 16))
        perf_s = self.small_font.render(f'FPS: {fps}', True, self.green_light)
        self.screen.blit(self._outlined(perf_s), (cx - r + 16, cy - r + 34))
        tracks_s = self.small_font.render(f'TRACKS: {self.last_track_count}', True, self.green_light)
        self.screen.blit(self._outlined(tracks_s), (cx - r + 16, cy - r + 52))
        # system health indicator (green/yellow/red)
        health_col = (0, 220, 100) if status == 'OK' else ((220, 180, 40) if status == 'DEGRADED' else (200, 60, 60))
        pygame.draw.circle(self.screen, health_col, (cx - r + 16 + 110, cy - r + 24), 6)

    def draw_sweep(self):
        cx, cy = self.center
        r = self.radius
        # motion-blurred sweep: draw a few recent sweep positions with decaying alpha
        steps = max(14, int(26 * 0.9))
        tail = math.radians(48.0)
        lead = math.radians(2.6)
        # draw micro blur behind current sweep
        for b in range(self.sweep_blur_steps, 0, -1):
            ang_offset = - (b * 0.006) * (1.0 + b * 0.1)  # small angular lag per blur layer
            ang = self.sweep_angle + ang_offset
            frac_tail = 0.0
            a1 = ang - tail
            a2 = a1 + max(lead, 0.001)
            alpha_mul = (self.sweep_blur_decay ** b)
            points = [(r, r)]
            segs = max(4, int(8))
            for s in range(segs + 1):
                angp = a1 + (a2 - a1) * (s / segs)
                px = r + math.cos(angp) * r
                py = r + math.sin(angp) * r
                points.append((px, py))
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            alpha = int(160 * alpha_mul)
            pygame.draw.polygon(surf, (self.green_light[0], self.green_light[1], self.green_light[2], alpha), points)
            self.screen.blit(surf, (cx - r, cy - r), special_flags=pygame.BLEND_ADD)
        # Bright leading edge - slightly softened and lower intensity
        lead_x = cx + math.cos(self.sweep_angle) * r
        lead_y = cy + math.sin(self.sweep_angle) * r
        pygame.draw.line(self.screen, (200, 230, 190), (cx, cy), (lead_x, lead_y), 2)
        glow_r = 14
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (self.green_light[0], self.green_light[1], self.green_light[2], 72), (glow_r, glow_r), glow_r)
        self.screen.blit(glow, (int(lead_x - glow_r), int(lead_y - glow_r)), special_flags=pygame.BLEND_ADD)

    def draw_tracks(self, tracks: List[object], frame_wh: Tuple[int, int], draw_trails: bool = True, draw_prediction: bool = True, draw_labels: bool = True):
        cx, cy = self.center
        W, H = frame_wh
        self.last_track_count = len(tracks)

        for t in tracks:
            sbbox = t.bbox if t.bbox else (int(t.cx - 2), int(t.cy - 2), 4, 4)
            angle, dist_norm = map_detection_to_radar(sbbox, W, H)
            rad = int(dist_norm * (self.radius - 22))
            sx = int(cx + math.cos(angle) * rad)
            sy = int(cy + math.sin(angle) * rad)

            # clamp inside scope
            if (sx - cx) ** 2 + (sy - cy) ** 2 > (self.radius - 6) ** 2:
                ang = math.atan2(sy - cy, sx - cx)
                sx = int(cx + math.cos(ang) * (self.radius - 6))
                sy = int(cy + math.sin(ang) * (self.radius - 6))

            # compute delta-time for smooth animations
            dt = min(0.1, max(1e-4, self._last_dt))

            # visual smoothing for track position (exponential ease)
            vis_k = 8.0  # responsiveness
            if not hasattr(t, '_vis_pos'):
                t._vis_pos = (sx, sy)
                t._vis_vel = (0.0, 0.0)
            vxp, vyp = t._vis_pos
            # lerp with time constant
            alpha = 1.0 - math.exp(-vis_k * dt)
            nx = vxp + (sx - vxp) * alpha
            ny = vyp + (sy - vyp) * alpha
            # subtle inertia: blend in a fraction of instantaneous velocity change
            inst_vx = (sx - vxp) / max(dt, 1e-4)
            inst_vy = (sy - vyp) / max(dt, 1e-4)
            t._vis_vel = (t._vis_vel[0] * 0.85 + inst_vx * 0.15, t._vis_vel[1] * 0.85 + inst_vy * 0.15)
            nx += t._vis_vel[0] * (0.02 * dt)
            ny += t._vis_vel[1] * (0.02 * dt)
            sx_vis = int(nx)
            sy_vis = int(ny)
            t._vis_pos = (nx, ny)

            # trails
            if draw_trails:
                n = len(t.trail)
                for idx, (txp, typ) in enumerate(list(t.trail)):
                    tbbox = (int(txp - 2), int(typ - 2), 4, 4)
                    ta, td = map_detection_to_radar(tbbox, W, H)
                    tr = int(td * (self.radius - 22))
                    tpx = int(cx + math.cos(ta) * tr)
                    tpy = int(cy + math.sin(ta) * tr)
                    if (tpx - cx) ** 2 + (tpy - cy) ** 2 > (self.radius - 6) ** 2:
                        continue
                    # distance-based fade
                    dist_factor = td
                    alpha_t = int(200 * ((idx + 1) / max(1, n)) * (0.4 + 0.6 * dist_factor))
                    surf = pygame.Surface((8, 8), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (self.green_light[0], self.green_light[1], self.green_light[2], alpha_t), (4, 4), max(1, int(3 * ((idx + 1) / max(1, n)))))
                    self.screen.blit(surf, (tpx - 4, tpy - 4))

            # uncertainty ellipse (based on confidence and stability)
            stab = t.stability()
            conf = t.conf
            # ellipse axes scale: lower conf/stab -> bigger ellipse
            base = 6
            major = int(base + (1.0 - conf) * 30 + (1.0 - stab) * 20)
            minor = int(base + (1.0 - conf) * 18 + (1.0 - stab) * 10)
            ell = pygame.Surface((major * 2, minor * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(ell, (200, 200, 200, 30), (0, 0, major * 2, minor * 2))
            # blit around visual position
            self.screen.blit(ell, (sx_vis - major, sy_vis - minor), special_flags=pygame.BLEND_PREMULTIPLIED)

            # halo glow
            score = t.threat_score((W, H))
            col = threat_color(score)
            # reduce bloom intensity for clarity
            halo_alpha = int(84 * (1.0 - stab))
            halo_radius = int(10 + (1.0 - stab) * 12)
            halo = pygame.Surface((halo_radius * 2, halo_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(halo, (col[0], col[1], col[2], halo_alpha), (halo_radius, halo_radius), halo_radius)
            self.screen.blit(halo, (sx_vis - halo_radius, sy_vis - halo_radius), special_flags=pygame.BLEND_PREMULTIPLIED)

            # diamond marker with pulse when sounds trigger (use visualized position)
            now = time.time()
            base_size = 8
            pulse_scale = 1.0
            if getattr(t, 'pulse_until', 0.0) > now:
                remaining = t.pulse_until - now
                pulse_scale = 1.0 + max(0.05, min(0.55, remaining * 3.0))
            # apply 3D tilt if enabled: slight vertical compression
            if getattr(self, 'enable_3d', False):
                tilt = 0.88
                sy_vis = int(cy + (sy_vis - cy) * tilt)
                rad = int(rad * 0.98)
            size = max(5, int(base_size * pulse_scale))
            pts = [(sx_vis, sy_vis - size), (sx_vis + size, sy_vis), (sx_vis, sy_vis + size), (sx_vis - size, sy_vis)]
            # draw vertical shadow below marker to imply height
            shadow = pygame.Surface((size * 3, size), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow, (0, 0, 0, 72), (0, 0, size * 3, size))
            self.screen.blit(shadow, (sx_vis - size, sy_vis + int(size * 0.6)), special_flags=pygame.BLEND_PREMULTIPLIED)
            pygame.draw.polygon(self.screen, col, pts)
            pygame.draw.polygon(self.screen, (20, 30, 20), pts, 1)
            # subtle extra ring when pulsing
            if pulse_scale > 1.02:
                ring = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
                ra = int(60 * min(1.0, pulse_scale - 1.0))
                pygame.draw.circle(ring, (col[0], col[1], col[2], ra), (size * 2, size * 2), int(size * 1.6), 2)
                self.screen.blit(ring, (sx_vis - size * 2, sy_vis - size * 2), special_flags=pygame.BLEND_ADD)

            # orientation notch
            vx, vy = t.vx, t.vy
            speed = math.hypot(vx, vy)
            if speed > 0.6:
                ang = math.atan2(vy, vx)
                notch_len = 6
                nx = sx_vis + int(math.cos(ang) * (size + notch_len))
                ny = sy_vis + int(math.sin(ang) * (size + notch_len))
                pygame.draw.circle(self.screen, (220, 230, 220), (nx, ny), 2)

            # prediction: dotted arc using smoothed velocity
            if draw_prediction:
                steps = 8
                for i in range(1, steps + 1):
                    frac = i / steps
                    # reduce prediction scale for visual cohesion and use vis pos
                    pxp = int(sx_vis + t.vx * (PREDICT_SEC * frac) * 0.045)
                    pyp = int(sy_vis + t.vy * (PREDICT_SEC * frac) * 0.045)
                    if (pxp - cx) ** 2 + (pyp - cy) ** 2 <= (self.radius - 6) ** 2:
                        dp = pygame.Surface((6, 6), pygame.SRCALPHA)
                        pygame.draw.circle(dp, (col[0], col[1], col[2], int(120 * (frac))), (3, 3), 2)
                        self.screen.blit(dp, (pxp - 3, pyp - 3))

            # labels (toggleable handled by caller) - use previous improved label code if draw_labels True
            if draw_labels:
                # Object name (larger) on first line, confidence/speed smaller on second line
                # use smoothed label position and alpha
                name = t.label
                meta = f"{int(t.conf*100)}%  {int(speed)}px/s"
                # append altitude label
                alt_lab = 'N/A'
                an = getattr(t, 'altitude_norm', None)
                if an is not None:
                    if an < 0.33:
                        alt_lab = 'ALT: LOW'
                    elif an < 0.66:
                        alt_lab = 'ALT: MID'
                    else:
                        alt_lab = 'ALT: HIGH'
                    meta = meta + '  ' + alt_lab

                # compute angular difference between sweep and target to fade labels when swept
                ang_diff = abs(((angle - self.sweep_angle + math.pi) % (2 * math.pi)) - math.pi)
                fade_norm = min(1.0, ang_diff / math.radians(12.0))  # 12 deg window
                text_fade = 0.6 + 0.4 * fade_norm  # when beam near -> darker (0.6), else full (1.0)

                name_color = (int(self.green_light[0] * text_fade), int(self.green_light[1] * text_fade), int(self.green_light[2] * text_fade))
                name_surf = self.font.render(name, True, name_color)
                meta_surf = self.small_font.render(meta, True, name_color)

                pad_x = 6
                pad_y = 4
                w = max(name_surf.get_width(), meta_surf.get_width()) + pad_x * 2
                h = name_surf.get_height() + meta_surf.get_height() + pad_y * 2 + 2

                ux = math.cos(angle)
                uy = math.sin(angle)
                perp_x, perp_y = -uy, ux
                side = -1 if math.cos(angle - self.sweep_angle) > 0 else 1
                offset_x = int((ux * 12) + (perp_x * 8 * side))
                offset_y = int((uy * 12) + (perp_y * 8 * side))

                # compute target label pos (based on visualized marker)
                lx_target = sx_vis + offset_x
                ly_target = sy_vis + offset_y - int(h / 2)
                # initialize visual label state
                if not hasattr(t, '_vis_label'):
                    t._vis_label = (lx_target, ly_target)
                    t._label_alpha = 1.0
                # smoothing for label glide
                lab_k = 8.0
                lab_alpha = 1.0 - math.exp(-lab_k * dt)
                lx = int(t._vis_label[0] + (lx_target - t._vis_label[0]) * lab_alpha)
                ly = int(t._vis_label[1] + (ly_target - t._vis_label[1]) * lab_alpha)
                t._vis_label = (lx, ly)
                # label fade when uncertain or commanded
                t._label_alpha += (1.0 - t._label_alpha) * lab_alpha if not getattr(t, 'uncertain', False) else (0.6 - t._label_alpha) * lab_alpha
                label_alpha_draw = max(0.35, min(1.0, t._label_alpha))

                # apply fade to plate and text
                plate = pygame.Surface((w, h), pygame.SRCALPHA)
                plate_color = (6, 18, 12, int(180 * text_fade * label_alpha_draw))
                pygame.draw.rect(plate, plate_color, (0, 0, w, h), border_radius=6)
                pygame.draw.rect(plate, (18, 60, 30, int(60 * text_fade * label_alpha_draw)), (0, 0, w, h), 1, border_radius=6)
                self.screen.blit(plate, (lx, ly), special_flags=pygame.BLEND_PREMULTIPLIED)

                # subtle outline via _outlined, but modulate alpha by converting surfaces
                name_s = self._outlined(name_surf).copy()
                meta_s = self._outlined(meta_surf).copy()
                # tint by label_alpha_draw using alpha modulation
                name_s.fill((255, 255, 255, int(255 * label_alpha_draw)), special_flags=pygame.BLEND_RGBA_MULT)
                meta_s.fill((255, 255, 255, int(255 * label_alpha_draw)), special_flags=pygame.BLEND_RGBA_MULT)
                self.screen.blit(name_s, (lx + pad_x, ly + pad_y))
                self.screen.blit(meta_s, (lx + pad_x, ly + pad_y + name_surf.get_height() + 2))

    # New UI panels and interaction helpers
    def draw_right_panel(self, tracks: List[object], selected_id: int, frame_wh: Tuple[int, int]):
        """Render vertical track details panel on the right side."""
        panel_w = 260
        cx, cy = self.center
        r = self.radius
        px = cx + r + 12
        py = cy - r
        ph = r * 2
        panel = pygame.Surface((panel_w, ph), pygame.SRCALPHA)
        # dark translucent background with subtle glow
        pygame.draw.rect(panel, (6, 18, 12, 220), (0, 0, panel_w, ph), border_radius=6)
        # header
        hdr = self.hud_font.render('TRACK DETAILS', True, self.green_light)
        panel.blit(hdr, (10, 8))
        # list top tracks (sorted by threat)
        sorted_tr = sorted(tracks, key=lambda t: t.threat_score(frame_wh), reverse=True)
        y = 36
        for t in sorted_tr[:12]:
            bg = (12, 30, 14) if t.id != selected_id else (20, 60, 30)
            pygame.draw.rect(panel, bg + (180,), (8, y - 2, panel_w - 16, 36), border_radius=4)
            # small ID block
            id_s = self.small_font.render(f'{t.label}:{t.id}', True, self.green_light)
            panel.blit(id_s, (14, y))
            # right side metrics
            conf_s = self.small_font.render(f'C:{int(t.conf*100)}%', True, self.green_light)
            speed = int(math.hypot(t.vx, t.vy))
            sp_s = self.small_font.render(f'S:{speed}', True, self.green_light)
            threat_s = self.small_font.render(f'T:{int(t.threat_score(frame_wh)*100)}%', True, self.green_light)
            panel.blit(conf_s, (panel_w - 80, y))
            panel.blit(sp_s, (panel_w - 80, y + 14))
            panel.blit(threat_s, (panel_w - 140, y + 14))
            y += 40
        self.screen.blit(panel, (px, py))

    def draw_bottom_log(self, events: deque):
        """Render bottom event log with fading older entries."""
        cx, cy = self.center
        r = self.radius
        log_h = 96
        lx = cx - r
        ly = cy + r + 12
        lw = r * 2
        log_s = pygame.Surface((lw, log_h), pygame.SRCALPHA)
        pygame.draw.rect(log_s, (4, 18, 12, 210), (0, 0, lw, log_h), border_radius=6)
        # draw recent events (newest at top)
        max_lines = 6
        for i, ev in enumerate(list(events)[:max_lines]):
            txt = f"{ev[0]} {ev[1]}"
            alpha = int(220 * (1.0 - i / max_lines))
            surf = self.small_font.render(txt, True, self.green_light)
            # shadow
            log_s.blit(self._outlined(surf), (8, 6 + i * 14))
        self.screen.blit(log_s, (lx, ly))

    def draw_left_elevation_strip(self, tracks: List[object], frame_h: int):
        """Render elevation abstraction strip on left side of radar.
        Tracks are shown as markers corresponding to bbox vertical position.
        """
        cx, cy = self.center
        r = self.radius
        strip_w = 28
        sx = cx - r - strip_w - 12
        sy = cy - r
        sh = r * 2
        strip = pygame.Surface((strip_w, sh), pygame.SRCALPHA)
        pygame.draw.rect(strip, (6, 18, 12, 200), (0, 0, strip_w, sh), border_radius=4)
        # draw layers
        layers = 8
        for i in range(layers):
            y = int(i * sh / layers)
            pygame.draw.line(strip, (10, 40, 20), (4, y), (strip_w - 4, y), 1)
        # show track markers
        for t in tracks:
            # elevation proxy: higher bbox center -> lower elevation
            _, by, bw, bh = t.bbox
            center_y = by + bh / 2.0
            elev_norm = 1.0 - (center_y / frame_h)
            my = int((1.0 - elev_norm) * sh)
            pygame.draw.circle(strip, self.green_light, (strip_w // 2, my), 4)
        self.screen.blit(strip, (sx, sy))

    def draw_mini_zoom(self, selected_track: object, tracks: List[object], frame_wh: Tuple[int, int]):
        """Draw a magnified mini-radar for the selected sector in the top-right corner.
        Shows nearby tracks relative to selected track.
        """
        if selected_track is None:
            return
        cx, cy = self.center
        r = self.radius
        mw = 180
        mh = 180
        mx = cx + r - mw - 12
        my = cy - r + 12
        mini = pygame.Surface((mw, mh), pygame.SRCALPHA)
        pygame.draw.circle(mini, (4, 18, 12, 220), (mw // 2, mh // 2), mw // 2)
        # center as selected track
        stx, sty = selected_track.cx, selected_track.cy
        # draw nearby tracks within sector radius
        for t in tracks:
            dx = t.cx - stx
            dy = t.cy - sty
            # scale down for mini
            sxm = int(mw // 2 + dx * 0.15)
            sym = int(mh // 2 + dy * 0.15)
            # ensure inside mini circle
            if (sxm - mw // 2) ** 2 + (sym - mh // 2) ** 2 > (mw // 2 - 6) ** 2:
                continue
            col = threat_color(t.threat_score(frame_wh))
            pygame.draw.circle(mini, col, (sxm, sym), 4)
        # highlight center
        pygame.draw.circle(mini, (200, 255, 200), (mw // 2, mh // 2), 6, 1)
        self.screen.blit(mini, (mx, my))

    # helper for panels
    def _outlined(self, surf: pygame.Surface) -> pygame.Surface:
        """Return rendered surface with a subtle dark outline for readability."""
        w, h = surf.get_size()
        out = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
        # dark offset for outline
        out.blit(surf, (1, 1))
        out.blit(surf, (0, 0))
        return out

    def draw_detections(self, dets: List[Tuple[str, float, float, float]]):
        cx, cy = self.center
        for label, angle, dist_norm, conf in dets:
            r = int(dist_norm * (self.radius - 22))
            px = int(cx + math.cos(angle) * r)
            py = int(cy + math.sin(angle) * r)
            if (px - cx) ** 2 + (py - cy) ** 2 > (self.radius - 6) ** 2:
                continue
            pygame.draw.circle(self.screen, (80, 180, 80), (px, py), 3)

    def flip(self):
        pygame.display.flip()

    def draw_depth_rings(self):
        cx, cy = self.center
        r = self.radius - 20
        # Colors for layers
        low_col = (6, 160, 80, 18)
        mid_col = (60, 200, 150, 14)
        high_col = (40, 150, 220, 10)
        # inner (LOW)
        pygame.draw.circle(self.screen, low_col, (cx, cy), int(r * 0.45))
        # mid
        pygame.draw.circle(self.screen, mid_col, (cx, cy), int(r * 0.72), 0)
        # outer (HIGH)
        pygame.draw.circle(self.screen, high_col, (cx, cy), int(r), 0)

    def draw_altitude_stack(self, tracks: List[object]):
        # vertical mini-strip showing altitude layers and track dots
        cx, cy = self.center
        r = self.radius
        strip_w = 40
        sx = cx - r - strip_w - 20
        sy = cy - r
        sh = r * 2
        s = pygame.Surface((strip_w, sh), pygame.SRCALPHA)
        pygame.draw.rect(s, (6, 18, 12, 200), (0, 0, strip_w, sh), border_radius=4)
        # draw three zones
        zones = [(0.0, 0.33, (6,160,80)), (0.33, 0.66, (60,200,150)), (0.66,1.0, (40,150,220))]
        for i, (a, b, col) in enumerate(zones):
            y0 = int((1.0 - b) * sh)
            y1 = int((1.0 - a) * sh)
            pygame.draw.rect(s, (col[0], col[1], col[2], 40), (0, y0, strip_w, max(2, y1 - y0)))
        # plot tracks as dots at altitude level
        for t in tracks:
            an = getattr(t, 'altitude_norm', 0.5)
            py = int(sy + sh * (1.0 - an))
            col = self._alt_color(an)
            pygame.draw.circle(s, col, (strip_w // 2, int(sh * (1.0 - an))), 5)
        self.screen.blit(s, (sx, sy))

    def _alt_color(self, an: float) -> Tuple[int, int, int]:
        # blend between low(mid->high) colors
        low = np.array([0,220,100])
        mid = np.array([60,200,150])
        high = np.array([40,150,220])
        if an < 0.33:
            t = an / 0.33
            col = low * (1 - t) + mid * t
        elif an < 0.66:
            t = (an - 0.33) / 0.33
            col = mid * (1 - t) + high * t
        else:
            t = (an - 0.66) / 0.34
            col = high * (1 - t) + np.array([80,220,200]) * t
        return (int(col[0]), int(col[1]), int(col[2]))


# ---------------------------------------------------------------------------
# Main application loop (enhanced)
# ---------------------------------------------------------------------------


def load_model() -> YOLO:
    try:
        model = YOLO("yolov8s.pt")
        return model
    except Exception as e:
        print("Failed to load YOLOv8s model:", e)
        raise


def main():
    print("Starting professional radar perception console. Press ESC to quit.")
    try:
        model = load_model()
    except Exception:
        print("Model load failed. Exiting.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    radar = RadarDisplay(SCREEN_SIZE)
    tracker = Tracker(MAX_TRACK_MISSED)
    last_t = time.time()

    # new features
    event_log = deque(maxlen=128)
    selected_track_id = -1
    replay_buffer = deque(maxlen=FPS * 30)
    replay_mode = False
    show_trails = True
    show_prediction = True
    show_labels = True
    enable_3d = False
    track_logs = []  # CSV rows
    detect_skip_counter = 0
    skip_factor = 1

    # --- audio setup -------------------------------------------------
    sound_soft = sound_tick = sound_warn = None
    mixer_ok = False
    try:
        pygame.mixer.init()
        mixer_ok = True
        import os
        
        # Resolve absolute path to assets/sounds relative to this script
        # This script is in doomsday/vision/, assets are in doomsday/assets/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) # doomsday/
        sounds_dir = os.path.join(project_root, 'assets', 'sounds')

        def _load(name, vol=0.06):
            path = os.path.join(sounds_dir, name)
            try:
                if not os.path.exists(path):
                    print(f"Sound file not found: {path}")
                    return None
                s = pygame.mixer.Sound(path)
                s.set_volume(vol)
                return s
            except Exception as e:
                print(f"Sound failed to load: {path} ({e})")
                return None
        sound_soft = _load('soft_beep.wav', vol=0.06)
        sound_tick = _load('tick.wav', vol=0.03)
        sound_warn = _load('warning.wav', vol=0.09)
    except Exception as e:
        print('Audio mixer init failed:', e)
        mixer_ok = False
    # -----------------------------------------------------------------

    try:
        running = True
        while running:
            now = time.time()
            dt = now - last_t
            last_t = now

            # adaptive skip: if FPS drops below target reduce detection frequency
            current_fps = radar.clock.get_fps() or FPS
            if current_fps < FPS * 0.85:
                skip_factor = min(4, skip_factor + 1)
            else:
                skip_factor = max(1, skip_factor - 1)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_t:
                        show_trails = not show_trails
                        event_log.append(("Toggled Trails", time.strftime('%H:%M:%S')))
                    elif ev.key == pygame.K_p:
                        show_prediction = not show_prediction
                        event_log.append(("Toggled Prediction", time.strftime('%H:%M:%S')))
                    elif ev.key == pygame.K_l:
                        show_labels = not show_labels
                        event_log.append(("Toggled Labels", time.strftime('%H:%M:%S')))
                    elif ev.key == pygame.K_r:
                        replay_mode = not replay_mode
                        event_log.append(("Replay Mode", f"ON" if replay_mode else "OFF"))
                    elif ev.key == pygame.K_3:
                        enable_3d = not enable_3d
                        radar.enable_3d = enable_3d
                        event_log.append(("3D Mode", "ON" if enable_3d else "OFF"))
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:  # left click
                    mx, my = ev.pos
                    for t in tracker.tracks:
                        sbbox = t.bbox if t.bbox else (int(t.cx - 2), int(t.cy - 2), 4, 4)
                        angle, dist_norm = map_detection_to_radar(sbbox, FRAME_WIDTH, FRAME_HEIGHT)
                        rad = int(dist_norm * (radar.radius - 22))
                        sx = int(radar.center[0] + math.cos(angle) * rad)
                        sy = int(radar.center[1] + math.sin(angle) * rad)
                        if (sx - mx) ** 2 + (sy - my) ** 2 <= 36:
                            selected_track_id = t.id
                            event_log.append(("Track Selected", time.strftime('%H:%M:%S') + f" ID:{t.id}"))

            # capture frame for replay (after drawing) when not in replay mode
            ret, frame = cap.read()
            detections = []
            do_detect = (detect_skip_counter % skip_factor) == 0
            detect_skip_counter += 1
            if ret and not replay_mode and do_detect:
                detections = detect_objects(model, frame)

            # update tracker every loop (even if detections empty)
            tracks = tracker.update(detections, now, (FRAME_WIDTH, FRAME_HEIGHT))

            # Sound triggers and pulses (per-track cooldowns)
            for t in tracks:
                now_t = time.time()
                # New track beep (play once shortly after creation)
                if getattr(t, 'created_at', 0) and (now_t - t.created_at) < 0.6 and (now_t - t.last_beep) > 1.5:
                    if mixer_ok and sound_soft:
                        try:
                            sound_soft.play()
                        except Exception:
                            pass
                    t.last_beep = now_t
                    t.pulse_until = now_t + 0.22

                # Sweep tick when beam passes over the target
                # compute angular difference
                try:
                    ang, _ = map_detection_to_radar(t.bbox, FRAME_WIDTH, FRAME_HEIGHT)
                except Exception:
                    ang = 0.0
                ang_diff = abs(((ang - radar.sweep_angle + math.pi) % (2 * math.pi)) - math.pi)
                if ang_diff < math.radians(2.5) and (now_t - t.last_tick) > 0.6:
                    if mixer_ok and sound_tick:
                        try:
                            sound_tick.play()
                        except Exception:
                            pass
                    t.last_tick = now_t
                    t.pulse_until = max(t.pulse_until, now_t + 0.08)

                # Warning tone: high confidence approaching target (radial velocity negative)
                if t.conf > 0.75 and (now_t - t.last_warning) > 3.0:
                    # radial velocity toward center of radar
                    rx = t.cx - radar.center[0]
                    ry = t.cy - radar.center[1]
                    rnorm = math.hypot(rx, ry)
                    if rnorm > 1.0:
                        radial_vel = (t.vx * rx + t.vy * ry) / rnorm
                        # approaching if radial_vel negative and magnitude significant
                        if radial_vel < -18.0:
                            if mixer_ok and sound_warn:
                                try:
                                    sound_warn.play()
                                except Exception:
                                    pass
                            t.last_warning = now_t
                            t.pulse_until = now_t + 0.4

            # logging: append per-track row (sampled)
            for t in tracks:
                track_logs.append((t.id, time.time(), t.cx, t.cy, math.hypot(t.vx, t.vy), t.conf, t.threat_score((FRAME_WIDTH, FRAME_HEIGHT))))

            # draw either live display or replay frame
            if replay_mode and len(replay_buffer) > 0:
                # cycle through stored surfaces
                surf = replay_buffer[0]
                self_surface = surf.copy()
                radar.screen.blit(self_surface, (0, 0))
            else:
                radar.clear()
                radar.draw_scope()
                radar.draw_sweep()
                radar.draw_tracks(tracks, (FRAME_WIDTH, FRAME_HEIGHT), draw_trails=show_trails, draw_prediction=show_prediction, draw_labels=show_labels)

                # UI panels
                radar.draw_right_panel(tracks, selected_track_id, (FRAME_WIDTH, FRAME_HEIGHT))
                radar.draw_bottom_log(event_log)
                radar.draw_left_elevation_strip(tracks, FRAME_HEIGHT)
                radar.draw_altitude_stack(tracks)
                radar.draw_mini_zoom(next((t for t in tracks if t.id == selected_track_id), None), tracks, (FRAME_WIDTH, FRAME_HEIGHT))

                if ret:
                    prev_h = 160
                    prev_w = int((frame.shape[1] / frame.shape[0]) * prev_h)
                    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_small = cv2.resize(preview, (prev_w, prev_h))
                    try:
                        surf = pygame.image.frombuffer(preview_small.tobytes(), (prev_w, prev_h), 'RGB')
                    except Exception:
                        surf = pygame.surfarray.make_surface(np.rot90(preview_small))
                    radar.screen.blit(surf, (10, SCREEN_SIZE[1] - prev_h - 10))

                # HUD small footer inside radar (sensor status already drawn in draw_scope)
                ops = radar.hud_font.render(f'Health: OK  Skip:{skip_factor}', True, RADAR_GREEN)
                radar.screen.blit(ops, (radar.center[0] - radar.radius + 16, radar.center[1] + radar.radius - 40))

                # store snapshot for replay
                if ret and not replay_mode:
                    replay_buffer.append(radar.screen.copy())

            radar.flip()
            radar.clock.tick(FPS)

    finally:
        # write CSV logs
        try:
            import csv
            csv_path = 'track_logs.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'time', 'x', 'y', 'speed', 'confidence', 'threat'])
                for row in track_logs:
                    writer.writerow(row)
            print(f'Track logs written to {csv_path}')
        except Exception as e:
            print('Failed to write CSV:', e)

        cap.release()
        pygame.quit()
        print('Shutdown complete.')


if __name__ == '__main__':
    main()
