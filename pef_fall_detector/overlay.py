"""Visual debugging overlay: skeleton, Step-0 anchors, and HUD.

Draws on plain OpenCV images so the same overlay works in the PEF-Lab GUI,
in exported PNG frames (paper figures), and in any future headless preview.
No Qt dependency here by design: the GUI is a viewer over the core, never
the other way around.

Colors follow one fixed convention so screenshots remain comparable across
the whole project:
    skeleton  – white          trunk vector    – yellow
    mid-hip   – red circle     mid-shoulder    – blue circle
    centroid  – magenta cross  HUD text        – green (ok) / red (warnings)
"""

from __future__ import annotations

import numpy as np

from .pose_frontend import PoseFrame

_WHITE = (255, 255, 255)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)
_GREEN = (0, 255, 0)
_MAGENTA = (255, 0, 255)


# MediaPipe Pose 33-landmark connection topology, declared locally so this
# module needs no MediaPipe import (recent releases dropped mp.solutions,
# where this constant used to live).
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm / hand
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm / hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg / foot
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg / foot
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


def _pose_connections() -> tuple[tuple[int, int], ...]:
    """Landmark connection topology used by the skeleton drawing."""
    return POSE_CONNECTIONS


def draw_overlay(
    frame_bgr: np.ndarray,
    pf: PoseFrame,
    visibility_threshold: float = 0.5,
    show_skeleton: bool = True,
    show_trunk: bool = True,
    show_centroid: bool = True,
    centroid_px: np.ndarray | None = None,
    hud_lines: list[str] | None = None,
) -> np.ndarray:
    """Draw the overlay onto a copy of ``frame_bgr`` and return it.

    Args:
        frame_bgr: source frame (not modified).
        pf: pose data for this frame.
        visibility_threshold: joints below this score are drawn dimmed.
        show_skeleton: toggle for the 33-landmark skeleton.
        show_trunk: toggle for the trunk vector (mid-hip → mid-shoulder),
            the geometric object behind Quantity T (§3.4).
        show_centroid: toggle for the smoothed centroid marker, the point
            whose vertical velocity is Quantity V (§3.4).
        centroid_px: smoothed centroid position from the pipeline (pixels);
            drawn only when provided and ``show_centroid`` is on.
        hud_lines: extra text lines appended to the on-screen panel
            (quantities, person state, warnings...).
    """
    import cv2

    out = frame_bgr.copy()

    if pf.detected:
        pts = pf.landmarks[:, :2].astype(int)

        if show_skeleton:
            for a, b in _pose_connections():
                dim = (
                    pf.visibility[a] < visibility_threshold
                    or pf.visibility[b] < visibility_threshold
                )
                color = (120, 120, 120) if dim else _WHITE
                cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 2)
            for i, (x, y) in enumerate(pts):
                dim = pf.visibility[i] < visibility_threshold
                cv2.circle(out, (x, y), 3, (120, 120, 120) if dim else _WHITE, -1)

        if show_trunk and pf.mid_hip is not None:
            hip = tuple(pf.mid_hip.astype(int))
            shoulder = tuple(pf.mid_shoulder.astype(int))
            cv2.line(out, hip, shoulder, _YELLOW, 3)
            cv2.circle(out, hip, 6, _RED, -1)
            cv2.circle(out, shoulder, 6, _BLUE, -1)

        if show_centroid and centroid_px is not None:
            cx, cy = int(centroid_px[0]), int(centroid_px[1])
            cv2.circle(out, (cx, cy), 8, _MAGENTA, 2)
            cv2.line(out, (cx - 12, cy), (cx + 12, cy), _MAGENTA, 2)
            cv2.line(out, (cx, cy - 12), (cx, cy + 12), _MAGENTA, 2)

    # --- HUD -----------------------------------------------------------------
    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f"frame {pf.frame_index}  t={pf.timestamp:6.2f}s", _GREEN))
    if pf.detected:
        lines.append((f"torso: {pf.torso_length:6.1f}px", _GREEN))
        vis_color = _GREEN if pf.core_visibility >= visibility_threshold else _RED
        lines.append((f"core visibility: {pf.core_visibility:.2f}", vis_color))
    else:
        lines.append(("NO PERSON DETECTED", _RED))
    for text in hud_lines or []:
        lines.append((text, _GREEN))

    y = 28
    for text, color in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28

    return out
