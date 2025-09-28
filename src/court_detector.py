# Optional TennisCourtDetector integration + simple fallback
# File: src/fullcourt/court_detector.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import sys
import cv2
import numpy as np

@dataclass
class CourtResult:
    overlay: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None               # homography img->plane (if available)
    px_to_m: Optional[float] = None              # average pixel-to-meter scale (rough)
    note: Optional[str] = None                   # caption for UI


class CourtDetector:
    """
    Tries to use yastrebksv/TennisCourtDetector if repo + weights are available.
    Otherwise draws a simple white-line overlay via Canny + HoughLinesP.
    """
    def __init__(self, enable_tcd: bool = True):
        self.enable_tcd = enable_tcd
        self.tcd_ready = False

        if enable_tcd:
            # Expect env vars:
            #   TCD_PATH: local path to cloned TennisCourtDetector repo
            #   TCD_WEIGHTS: path to downloaded model weights (*.pth from README)
            self.tcd_path = os.environ.get("TCD_PATH")
            self.tcd_weights = os.environ.get("TCD_WEIGHTS")
            if self.tcd_path and os.path.isdir(self.tcd_path) and self.tcd_weights and os.path.isfile(self.tcd_weights):
                # Add repo root to sys.path so we can import its modules (they live at repo root)
                if self.tcd_path not in sys.path:
                    sys.path.append(self.tcd_path)
                try:
                    # Lazy imports; we only check availability here.
                    import tracknet  # noqa: F401
                    import postprocess  # noqa: F401
                    import homography  # noqa: F401
                    import court_reference  # noqa: F401
                    self.tcd_ready = True
                except Exception:
                    self.tcd_ready = False

    def infer(self, frame: np.ndarray) -> CourtResult:
        if self.enable_tcd and self.tcd_ready:
            try:
                # --- Minimal, robust integration path ---
                # We call the repo's video/image pipeline concepts:
                # 1) Use 'tracknet' to predict 14+1 keypoints (heatmap-based)
                # 2) 'postprocess' to refine & 'homography' to compute image->court-plane transform
                #    (The README describes 14 keypoints + center and homography refinement.)
                # 3) Estimate a rough px->m scale from court reference geometry.
                import torch
                import tracknet
                import postprocess
                import homography
                import court_reference

                device = torch.device("cpu")
                model = tracknet.TrackNet()  # model architecture defined in repo
                model.load_state_dict(torch.load(self.tcd_weights, map_location=device))
                model.to(device).eval()

                # Prepare input to 640x360 as per README
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h0, w0 = img.shape[:2]
                resized = cv2.resize(img, (640, 360))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float()[None] / 255.0

                with torch.no_grad():
                    pred = model(tensor).cpu().numpy()[0]  # heatmaps

                # Postprocess → get keypoints in the resized space, then map back to original image
                kps_360 = postprocess.get_keypoints_from_heatmaps(pred)  # expected API in repo
                scale_x = w0 / 640.0
                scale_y = h0 / 360.0
                kps_img = [(int(x * scale_x), int(y * scale_y)) if x is not None else None for (x, y) in kps_360]

                # Homography + overlay (APIs per repo modules)
                ref = court_reference.get_reference_keypoints()  # reference plane coordinates
                H = homography.estimate_homography(kps_img, ref)
                overlay = self._draw_court_from_keypoints(frame, kps_img)

                # Rough px->m scale: use singles court length 23.77m if we can map two far points
                px_to_m = None
                try:
                    # Choose two opposite baseline center points from keypoints if defined
                    # (Exact indices depend on repo; we guard with try/except.)
                    iA, iB = 0, 7  # placeholder indices
                    if kps_img[iA] and kps_img[iB]:
                        dpx = np.hypot(kps_img[iA][0] - kps_img[iB][0], kps_img[iA][1] - kps_img[iB][1])
                        if dpx > 10:
                            px_to_m = 23.77 / dpx
                except Exception:
                    pass

                return CourtResult(overlay=overlay, H=H, px_to_m=px_to_m,
                                   note="Court: TennisCourtDetector model (homography enabled)")
            except Exception:
                # If anything goes wrong, gracefully fall back
                pass

        # --- Fallback: simple white-line overlay with Canny/Hough ---
        edges = cv2.Canny(cv2.GaussianBlur(frame, (5, 5), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=100, maxLineGap=15)
        overlay = frame.copy()
        if lines is not None:
            for l in lines[:, 0]:
                cv2.line(overlay, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 255, 255), 2)
            note = "Court: edge/Hough fallback (no homography)"
        else:
            note = "Court: not detected"
        return CourtResult(overlay=overlay, H=None, px_to_m=None, note=note)

    def draw(self, frame: np.ndarray, result: CourtResult) -> np.ndarray:
        if result and result.overlay is not None:
            # Blend overlay 60%
            return cv2.addWeighted(result.overlay, 0.6, frame, 0.4, 0)
        return frame

    @staticmethod
    def _draw_court_from_keypoints(frame: np.ndarray, kps: list) -> np.ndarray:
        out = frame.copy()
        color = (255, 255, 255)
        # Connect plausible court segments if indices exist; this is robust to missing kps
        def line(i, j):
            if i < len(kps) and j < len(kps) and kps[i] and kps[j]:
                cv2.line(out, kps[i], kps[j], color, 2)
        # Skeleton (indices here are placeholders; adapt if you refine keypoint ordering)
        pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),  # perimeter-ish
                 (0,8),(7,9),(1,10),(6,11),(2,12),(5,13)]
        for i, j in pairs:
            line(i, j)
        return out
