# File: src/pages/2_Full_Court_Analysis.py
from __future__ import annotations

import os
import glob
import pathlib
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from ultralytics import YOLO

# =========================
# Utilities: paths & video
# =========================

def resolve_video_dir(user_input: str) -> Path:
    """Return an absolute folder path. Tries CWD, project root (parent of src), and ~ expansion."""
    p = Path(user_input).expanduser()
    if p.is_absolute() and p.exists():
        return p

    # project root = parent of /.../src/pages/2_Full_Court_Analysis.py
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / user_input,
        project_root / user_input,
        project_root / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd() / user_input  # show where we're looking even if it doesn't exist


def collect_videos(folder: Path) -> List[str]:
    exts = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV")
    vids: List[Path] = []
    for e in exts:
        vids.extend(folder.glob(e))
    return sorted(v.name for v in vids)


# =========================
# Trackers (BoxMOT -> fallback)
# =========================

def _make_tracker():
    """
    Try BoxMOT's tracker zoo (ByteTrack). If the YAML isn't present (common on fresh installs),
    fall back to a simple passthrough 'tracker' that just assigns pseudo-IDs.
    """
    try:
        from boxmot.tracker_zoo import create_tracker
        import boxmot  # to locate configs directory
        root = pathlib.Path(boxmot.__file__).resolve().parent
        cfg_path = root / "configs" / "bytetrack.yaml"
        return create_tracker("bytetrack", tracker_config=str(cfg_path), device="cpu", half=False)
    except Exception as e:
        st.warning(
            f"BoxMOT tracker config not found or failed ({e}). Using a simple fallback tracker."
        )

        class _Fallback:
            _next_id: int = 1

            def update(self, dets, frame):
                """
                dets: Nx6 [x1,y1,x2,y2,conf,cls]
                return Nx8 [x1,y1,x2,y2,id,conf,cls,ind]
                """
                if dets is None or len(dets) == 0:
                    return np.empty((0, 8), dtype=np.float32)
                out = []
                for i, d in enumerate(dets):
                    x1, y1, x2, y2 = d[:4]
                    conf = d[4] if len(d) > 4 else 1.0
                    cls = d[5] if len(d) > 5 else -1
                    out.append([
                        float(x1), float(y1), float(x2), float(y2),
                        float(self._next_id + i), float(conf), float(cls), 0.0
                    ])
                self._next_id += len(dets)
                return np.asarray(out, dtype=np.float32)

        return _Fallback()


# =========================
# Optional court detector
# =========================
try:
    from fullcourt.court_detector import CourtDetector, CourtResult  # your wrapper (optional)
except Exception:
    CourtDetector = None
    CourtResult = None  # type: ignore


# =========================
# Helpers & geometry
# =========================

COCO_PERSON = 0
COCO_SPORTS_BALL = 32
SINGLES_WIDTH_M = 8.23  # official singles width

def xyxy_to_xywh(x1, y1, x2, y2) -> Tuple[float, float, float, float]:
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def safe_court_mask(frame: np.ndarray, result: Optional["CourtResult"]) -> np.ndarray:
    """
    Return a binary (uint8) mask of the playable court.
    If no polygon available, return a conservative rectangle occupying most of the frame.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    try:
        poly = getattr(result, "polygon", None)
        if poly is not None:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
        else:
            cv2.rectangle(
                mask,
                (int(0.08 * w), int(0.18 * h)),
                (int(0.92 * w), int(0.92 * h)),
                255, -1
            )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    except Exception:
        cv2.rectangle(
            mask,
            (int(0.10 * w), int(0.20 * h)),
            (int(0.90 * w), int(0.90 * h)),
            255, -1
        )
    return mask


def draw_court_outline_safe(img: np.ndarray, mask: np.ndarray, net_y: int) -> None:
    """
    Draw a clean court outline + net line from a binary mask, per-frame (no caching).
    Avoids the 'first-frame overlay' artifact.
    """
    if mask is None:
        return
    m = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if len(c) >= 3:
            cv2.polylines(img, [c], True, (230, 230, 230), 2, lineType=cv2.LINE_AA)
    net_y = int(np.clip(net_y, 0, img.shape[0] - 1))
    cv2.line(img, (0, net_y), (img.shape[1] - 1, net_y), (230, 230, 230), 1, lineType=cv2.LINE_AA)


def estimate_scale_from_mask(mask: np.ndarray, y_row: int) -> Optional[float]:
    """
    Estimate meters-per-pixel at a given row by measuring court width in the mask
    and dividing official singles width (8.23 m) by that pixel width.
    """
    if mask is None:
        return None
    m = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    H, W = m.shape[:2]
    y = int(np.clip(y_row, 0, H - 1))
    row = m[y, :]

    xs = np.where(row > 0)[0]
    if xs.size < 2:
        return None
    width_px = float(xs.max() - xs.min())
    if width_px <= 2:
        return None
    return SINGLES_WIDTH_M / width_px


# =========================
# Player filtering (only 2 on-court players)
# =========================

def filter_people(tracks, mask: np.ndarray, h_img: int, net_y: int) -> List[Dict[str, Any]]:
    """
    Keep only true players:
      • footpoint inside court mask
      • exclude chair-umpire / sidelines
      • perspective-aware size/aspect constraints near vs far half
    """
    out: List[Dict[str, Any]] = []
    if tracks is None or getattr(tracks, "xyxy", None) is None or len(tracks.xyxy) == 0:
        return out

    xyxy = tracks.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    else:
        xyxy = np.asarray(xyxy)

    tids = getattr(tracks, "tracker_id", None)
    if tids is None:
        tids = np.full((len(xyxy),), -1, dtype=int)
    else:
        tids = np.asarray(tids)
        if hasattr(tids, "cpu"):
            tids = tids.cpu().numpy()
        tids = tids.astype(int)

    work = mask if mask.dtype == np.uint8 else mask.astype(np.uint8)
    H, W = work.shape[:2]
    # suppress extreme sides (ball kids / line judges)
    margin = int(0.10 * W)
    work_side = work.copy()
    work_side[:, :margin] = 0
    work_side[:, W - margin :] = 0

    # suppress typical chair umpire zone (broadcast right)
    cx1 = int(0.76 * W)
    cy1 = int(net_y - 0.33 * H)
    cx2 = W
    cy2 = int(net_y + 0.36 * H)
    cv2.rectangle(work_side, (cx1, max(0, cy1)), (cx2, min(H - 1, cy2)), 0, -1)

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        tid = int(tids[i]) if i < len(tids) else -1
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue

        # footpoint must be playable
        foot_x = (x1 + x2) / 2.0
        foot_y = y2
        fx = int(np.clip(foot_x, 0, W - 1))
        fy = int(np.clip(foot_y, 0, H - 1))
        if work_side[fy, fx] == 0:
            continue

        # perspective-aware gate
        if foot_y < net_y:  # far half
            min_h_frac = 0.040
            ar_min, ar_max = 0.65, 7.0
        else:  # near half
            min_h_frac = 0.095
            ar_min, ar_max = 0.90, 6.0

        if h < h_img * min_h_frac:
            continue
        ar = h / max(w, 1e-6)
        if not (ar_min <= ar <= ar_max):
            continue

        out.append({
            "id": tid,
            "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "foot_x": foot_x,
            "foot_y": foot_y,
            "h": h,
        })

    # Keep at most two biggest by height
    out.sort(key=lambda d: d["h"], reverse=True)
    return out[:2]


# =========================
# Player kinematics (with smoothing)
# =========================

@dataclass
class PlayerKinematics:
    last_t: Optional[float] = None
    last_pos: Optional[Tuple[float, float]] = None  # (cx, cy) in px
    total_dist_m: float = 0.0
    speeds_mps: List[float] = field(default_factory=list)
    path_px: List[Tuple[int, int]] = field(default_factory=list)
    ema_pos: Optional[Tuple[float, float]] = None  # exponential moving average for pos
    ema_alpha: float = 0.3
    max_jump_px: float = 120.0  # gate spikes

    def reset(self):
        self.last_t = None
        self.last_pos = None
        self.total_dist_m = 0.0
        self.speeds_mps.clear()
        self.path_px.clear()
        self.ema_pos = None

    def update(self, t: float, cx: float, cy: float, m_per_px: Optional[float]):
        # position EMA smoothing
        if self.ema_pos is None:
            self.ema_pos = (cx, cy)
        else:
            ex, ey = self.ema_pos
            self.ema_pos = (ex + self.ema_alpha * (cx - ex), ey + self.ema_alpha * (cy - ey))
        sx, sy = self.ema_pos

        self.path_px.append((int(sx), int(sy)))
        if len(self.path_px) > 300:
            self.path_px.pop(0)

        if self.last_t is not None and self.last_pos is not None:
            dt = max(t - self.last_t, 1e-6)
            dx = sx - self.last_pos[0]
            dy = sy - self.last_pos[1]
            px_dist = float(np.hypot(dx, dy))
            # gate unrealistic jumps (ID swap)
            if px_dist < self.max_jump_px and m_per_px:
                d_m = px_dist * m_per_px
                v_mps = d_m / dt
                self.total_dist_m += d_m
                # clamp absurd values (rare)
                if 0.0 <= v_mps < 12.0:  # ~43 km/h running is extreme
                    self.speeds_mps.append(v_mps)

        self.last_t = t
        self.last_pos = (sx, sy)

    def summary(self) -> Dict[str, float]:
        if not self.speeds_mps:
            return {"avg_mps": 0.0, "top_mps": 0.0, "dist_m": self.total_dist_m}
        avg = float(np.mean(self.speeds_mps))
        top = float(np.max(self.speeds_mps))
        return {"avg_mps": avg, "top_mps": top, "dist_m": self.total_dist_m}


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Full Court Analysis", page_icon="🎾", layout="wide")
st.title("🎾 Full Court Analysis (broadcast view: 2 players + ball + court)")

with st.sidebar:
    st.header("Settings")
    data_dir_in = st.text_input("Video folder", value="data")
    yolo_model = st.text_input("YOLO model", value="yolov8n.pt")
    conf_thres = st.slider("Detection confidence", 0.05, 0.75, 0.35, 0.05)
    stride = st.slider("Frame stride (analyze every Nth frame)", 1, 5, 2, 1)
    show_trails = st.checkbox("Show ball trail", value=True)
    show_ids = st.checkbox("Show player IDs", value=False)
    use_tcd = st.checkbox("Use TennisCourtDetector (if available)", value=True)

    st.markdown("**Rally detection (ball motion)**")
    px_thresh = st.slider("Ball speed threshold (px/s)", 80, 800, 220, 10,
                          help="Rally starts when ball speed exceeds this for 3 frames; ends after 10 still frames.")
    force_rally = st.checkbox("Force rally (always on)", value=False)

    run_btn = st.button("▶️ Run full-court analysis", type="primary", use_container_width=True)

videos_dir = resolve_video_dir(data_dir_in)
videos = collect_videos(videos_dir)

if not videos:
    st.info(
        f"No videos found in **{videos_dir}**.\n\n"
        "Supported extensions: .AVI, .MKV, .MOV, .MP4 (any case). "
        "Drop a broadcast-angle clip in that folder or change the path in the sidebar."
    )
    st.stop()

video_name = st.selectbox("Pick a video", videos, index=0, key="fullcourt_video")
st.caption(f"Looking in: {videos_dir}")

col_vid, col_side = st.columns([2.3, 1])
video_area = col_vid.empty()
stats_box = col_side.container()
charts_box = col_side.container()
notes_box = col_side.container()


# =========================
# Main
# =========================

if run_btn:
    path = str(videos_dir / video_name)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("Could not open the selected video.")
        st.stop()

    with st.spinner("Loading models…"):
        det = YOLO(yolo_model)  # COCO model (0=person, 32=sports ball)
        person_tracker = _make_tracker()
        ball_tracker = _make_tracker()
        court = CourtDetector(enable_tcd=use_tcd) if CourtDetector else None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    last_ball_center: Optional[Tuple[float, float]] = None
    last_ball_time: Optional[float] = None
    ball_trail: List[Tuple[int, int]] = []

    court_result: Optional["CourtResult"] = None
    court_mask: Optional[np.ndarray] = None
    net_y_guess: Optional[int] = None

    # Rally state
    rally_active = False
    still_cnt = 0
    move_cnt = 0

    # Two trackers: chart (always-on) and rally (reset per rally)
    p1_all = PlayerKinematics()    # always-on for chart
    p1_rally = PlayerKinematics()  # per-rally stats
    p1_speed_series: List[Tuple[float, float]] = []  # (t, m/s), always-on
    p1_id: Optional[int] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride:
            continue

        t = frame_idx / fps
        H, W = frame.shape[:2]

        # Detect persons + ball
        pred = det.predict(
            frame, conf=conf_thres, classes=[COCO_PERSON, COCO_SPORTS_BALL],
            verbose=False, device="cpu"
        )[0]

        if pred.boxes is not None and pred.boxes.data is not None:
            xyxy = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy()
            dets = np.concatenate([xyxy, confs[:, None], clss[:, None]], axis=1).astype(np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        persons_mask = dets[:, -1] == COCO_PERSON if dets.size else np.array([], bool)
        balls_mask = dets[:, -1] == COCO_SPORTS_BALL if dets.size else np.array([], bool)
        dets_person = dets[persons_mask] if dets.size else dets
        dets_ball = dets[balls_mask] if dets.size else dets

        tr_persons = person_tracker.update(dets_person, frame)
        tr_balls = ball_tracker.update(dets_ball, frame)

        # Court once
        if court_result is None and court:
            try:
                court_result = court.infer(frame)
            except Exception:
                court_result = None
        if court_mask is None:
            court_mask = safe_court_mask(frame, court_result)
        if net_y_guess is None:
            net_y_guess = int(getattr(court_result, "net_y", H // 2))

        # Filter true players
        class _Tracks:
            def __init__(self, arr: np.ndarray):
                if arr is None or len(arr) == 0:
                    self.xyxy = np.empty((0, 4), np.float32)
                    self.tracker_id = np.empty((0,), np.int32)
                else:
                    self.xyxy = arr[:, :4]
                    self.tracker_id = arr[:, 4].astype(int)

        candidates = filter_people(_Tracks(tr_persons), court_mask, H, net_y_guess)

        # Choose Player 1 = near-half player; keep ID consistency if possible
        p1 = None
        near = [p for p in candidates if p["foot_y"] >= net_y_guess]
        if p1_id is not None:
            for p in near:
                if p["id"] == p1_id:
                    p1 = p
                    break
        if p1 is None and near:
            p1 = max(near, key=lambda d: d["h"])
            p1_id = p1["id"]

        # Ball center & instantaneous px/s (for rally gating)
        disp = frame.copy()
        ball_center = None
        ball_px_s = None
        for tb in tr_balls:
            x1, y1, x2, y2 = tb[:4].astype(float)
            cx, cy, *_ = xyxy_to_xywh(x1, y1, x2, y2)
            ball_center = (cx, cy)
            if show_trails:
                ball_trail.append((int(cx), int(cy)))
                if len(ball_trail) > 120:
                    ball_trail.pop(0)
                for i in range(1, len(ball_trail)):
                    cv2.line(disp, ball_trail[i - 1], ball_trail[i], (0, 255, 0), 2)
            cv2.circle(disp, (int(cx), int(cy)), 6, (0, 255, 0), 2)

            if last_ball_center is not None and last_ball_time is not None:
                dtb = max(t - last_ball_time, 1.0 / fps)
                ball_px_s = float(np.hypot(cx - last_ball_center[0], cy - last_ball_center[1]) / dtb)
            last_ball_center, last_ball_time = (cx, cy), t
            break  # one ball max

        # Rally segmentation
        if force_rally:
            rally_active = True
        else:
            moving = ball_px_s is not None and ball_px_s > px_thresh
            if moving:
                move_cnt += 1
                still_cnt = 0
            else:
                still_cnt += 1
                move_cnt = 0

            if not rally_active and move_cnt >= 3:
                rally_active = True
                p1_rally.reset()

            if rally_active and still_cnt >= 10:
                rally_active = False
                # keep last summary for sidebar
                # (computed later when showing stats)

        # Meters-per-pixel: prefer TCD scale else estimate from mask width at player row
        m_per_px = getattr(court_result, "px_to_m", None) if court_result else None
        if m_per_px is None and p1 is not None:
            m_per_px = estimate_scale_from_mask(court_mask, int(p1["foot_y"]))

        # Update kinematics
        if p1 is not None:
            cx, cy = p1["cx"], p1["cy"]
            p1_all.update(t, cx, cy, m_per_px)
            if p1_all.speeds_mps:
                p1_speed_series.append((t, p1_all.speeds_mps[-1]))
                if len(p1_speed_series) > 1500:
                    p1_speed_series = p1_speed_series[-1500:]
            if rally_active:
                p1_rally.update(t, cx, cy, m_per_px)

        # Draw player boxes
        for p in candidates:
            x1, y1, x2, y2 = p["xyxy"]
            color = (255, 200, 0)
            cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if show_ids:
                cv2.putText(disp, f"ID {p['id']}", (int(x1), max(20, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

        if p1 is not None:
            x1, y1, x2, y2 = p1["xyxy"]
            cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
            cv2.putText(disp, "Player 1", (int(x1), max(20, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Player 1 path (green)
        if len(p1_all.path_px) > 1:
            for i in range(1, len(p1_all.path_px)):
                cv2.line(disp, p1_all.path_px[i - 1], p1_all.path_px[i], (0, 255, 160), 2)

        # Court overlay (no first-frame rectangle)
        has_polygon = (court_result is not None) and (getattr(court_result, "polygon", None) is not None)
        if has_polygon and court_mask is not None and net_y_guess is not None:
            draw_court_outline_safe(disp, court_mask, net_y_guess)
        else:
            # Draw only the net line to avoid the big white rectangle
            if net_y_guess is not None:
                cv2.line(disp, (0, int(net_y_guess)), (W - 1, int(net_y_guess)),
                         (230, 230, 230), 1, lineType=cv2.LINE_AA)

        # Stream current frame
        video_area.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()

    # ================= Stats & Charts =================
    # Rally summary from p1_rally
    rally_sum = p1_rally.summary()
    avg_mps = rally_sum["avg_mps"]
    top_mps = rally_sum["top_mps"]
    dist_m = rally_sum["dist_m"]
    avg_fts = avg_mps * 3.28084
    top_fts = top_mps * 3.28084

    with stats_box:
        st.subheader("Player 1 — Rally kinematics")
        st.write(
            f"**Speed:** Avg {avg_mps:.2f} m/s ({avg_fts:.2f} ft/s) · "
            f"Top {top_mps:.2f} m/s ({top_fts:.2f} ft/s)"
        )
        st.write(f"**Distance:** {dist_m:.2f} m")

        if (court_result is None) or (getattr(court_result, 'px_to_m', None) is None):
            st.caption(
                "Scale estimated from **official singles width** (8.23 m) at the player’s row. "
                "Connect TennisCourtDetector for exact homography-based meters."
            )

    with charts_box:
        st.subheader("Player 1 — Speed over time (m/s)")
        if p1_speed_series:
            df = pd.DataFrame([{"t": t, "mps": v} for (t, v) in p1_speed_series])
            st.line_chart(df.set_index("t")["mps"], height=180)
        else:
            st.caption("No samples captured yet — waiting for Player 1 to be detected.")

    with notes_box:
        st.markdown("---")
        st.caption(
            "Rally detection uses ball motion (> threshold) sustained for 3 frames to start, "
            "and 10 still frames to end. "
            "Only on-court players are kept (sidelines / chair area suppressed). "
            "Court overlay draws a net line unless a polygon is available."
        )
