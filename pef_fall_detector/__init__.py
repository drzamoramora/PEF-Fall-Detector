"""PEF-Fall-Detector.

Physics-informed, explainable, markerless fall detection from MediaPipe
pose landmarks. Reference implementation for the paper:

    "Beyond Black-Box AI: A Physics-Informed Edge Framework for Explainable
    Markerless Fall Detection Using MediaPipe Landmarks" (ULACIT).

Package layout (mirrors the paper's architecture, §3.1):
    sources        frame acquisition (video file or live camera) — source-agnostic core
    pose_frontend  MediaPipe landmark extraction + Step-0 normalization (§3.2)
    quantities     the four physical quantities T, V, P, I (§3.4)      [Phase 2+]
    state_machine  three-stage confirmation logic (§3.5)               [Phase 3+]
    audit_log      per-frame CSV that makes every decision auditable (§3.5)
    overlay        visual debugging overlay (skeleton, HUD)
    gui            PEF-Lab: PySide6 visualization workbench
"""

__version__ = "0.1.0"
