"""Parametric OpenPose skeletons for ControlNet conditioning.

CLIP-guided SDXL cannot reliably hold a pose or place a subject "on top of" something from text.
ControlNet fixes this by conditioning on an OpenPose skeleton — but a *fixed* skeleton only ever
draws the one pose it was hand-built for. Here the skeleton is built **parametrically** from the
joint angles the scene planner derived from the prompt (see ``scene_planner``), so any pose the
LLM can describe is renderable. Nothing pose-specific is hardcoded: there is a neutral rig plus
forward kinematics, and the angles come entirely from the plan.

Keypoints follow the COCO-18 OpenPose convention; limbs use the canonical OpenPose colours the
SDXL OpenPose ControlNet was trained on.

Angle convention (degrees), matching ``scene_planner._POSE_HINT``:
  shoulder/hip: 0 = limb straight down, 90 = straight out to the side, 180 = straight up.
  elbow/knee:   flex, 0 = straight, 90 = right angle.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

# COCO-18 keypoint indices:
#  0 nose 1 neck 2 r_shoulder 3 r_elbow 4 r_wrist 5 l_shoulder 6 l_elbow 7 l_wrist
#  8 r_hip 9 r_knee 10 r_ankle 11 l_hip 12 l_knee 13 l_ankle 14 r_eye 15 l_eye 16 r_ear 17 l_ear
_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# Neutral rig bone lengths in arbitrary local units (down = +y). The skeleton is built in this
# local space, then uniformly scaled/translated to fit the entity box, so raised arms that
# extend past the head are still contained.
_HEAD = 0.18          # neck -> nose
_SHOULDER_HALF = 0.18
_UPPER_ARM = 0.30
_FOREARM = 0.26
_TORSO = 0.55         # neck -> hip centre
_HIP_HALF = 0.11
_THIGH = 0.45
_SHIN = 0.42

# Per-stance default joint angles; the planner's joints overlay these. Defaults alone already
# distinguish e.g. standing (legs extended) from sitting (thighs/knees folded).
_STANCE_DEFAULTS = {
    "standing":  {"shoulder": 18, "elbow": 8, "hip": 8, "knee": 4},
    "sitting":   {"shoulder": 25, "elbow": 25, "hip": 88, "knee": 88},
    "crouching": {"shoulder": 30, "elbow": 40, "hip": 105, "knee": 115},
    "lying":     {"shoulder": 80, "elbow": 10, "hip": 85, "knee": 10},
}


def _limb_end(anchor: tuple[float, float], angle_deg: float, length: float,
              side: int) -> tuple[float, float]:
    """End point of a bone from ``anchor`` at ``angle_deg`` from straight-down, on ``side``
    (+1 = subject-left / viewer-right, -1 = subject-right)."""
    a = math.radians(angle_deg)
    return (anchor[0] + side * math.sin(a) * length, anchor[1] + math.cos(a) * length)


def _joint(spec_joints: dict, name: str, default: float) -> float:
    try:
        return float(spec_joints.get(name, default))
    except (TypeError, ValueError):
        return default


def _local_keypoints(pose_spec: dict) -> list[tuple[float, float]]:
    """Build the 18 COCO keypoints in local units via forward kinematics."""
    stance = str(pose_spec.get("stance", "standing")).lower()
    d = _STANCE_DEFAULTS.get(stance, _STANCE_DEFAULTS["standing"])
    j = pose_spec.get("joints") if isinstance(pose_spec.get("joints"), dict) else {}

    neck = (0.0, 0.0)
    nose = (0.0, -_HEAD)
    r_sh = (-_SHOULDER_HALF, 0.03)
    l_sh = (_SHOULDER_HALF, 0.03)
    hip_c = (0.0, _TORSO)
    r_hip = (-_HIP_HALF, _TORSO)
    l_hip = (_HIP_HALF, _TORSO)

    def arm(shoulder, side, sh_name, el_name):
        sh = _joint(j, sh_name, d["shoulder"])
        el = _joint(j, el_name, d["elbow"])
        elbow = _limb_end(shoulder, sh, _UPPER_ARM, side)
        wrist = _limb_end(elbow, sh - el, _FOREARM, side)  # bend toward the up direction
        return elbow, wrist

    def leg(hip, side, hip_name, knee_name):
        hp = _joint(j, hip_name, d["hip"])
        kn = _joint(j, knee_name, d["knee"])
        knee = _limb_end(hip, hp, _THIGH, side)
        ankle = _limb_end(knee, hp + kn, _SHIN, side)  # knee flexes forward
        return knee, ankle

    r_el, r_wr = arm(r_sh, -1, "r_shoulder_deg", "r_elbow_deg")
    l_el, l_wr = arm(l_sh, +1, "l_shoulder_deg", "l_elbow_deg")
    r_kn, r_an = leg(r_hip, -1, "r_hip_deg", "r_knee_deg")
    l_kn, l_an = leg(l_hip, +1, "l_hip_deg", "l_knee_deg")

    return [
        nose, neck, r_sh, r_el, r_wr, l_sh, l_el, l_wr,
        r_hip, r_kn, r_an, l_hip, l_kn, l_an,
        (-0.045, -_HEAD - 0.02), (0.045, -_HEAD - 0.02),  # eyes
        (-0.09, -_HEAD + 0.01), (0.09, -_HEAD + 0.01),    # ears
    ]


def _fit_to_box(pts: list[tuple[float, float]], box: list[float], width: int,
                height: int) -> list[tuple[float, float]]:
    """Uniformly scale/translate local keypoints to fill the entity box; feet at the box
    bottom, horizontally centred — so a 'standing on X' figure's feet land on X's top edge."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    bw, bh = (maxx - minx) or 1e-6, (maxy - miny) or 1e-6

    px0, py0, px1, py1 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    box_w, box_h = (px1 - px0) or 1.0, (py1 - py0) or 1.0
    scale = min(box_w / bw, box_h / bh)

    drawn_w, drawn_h = bw * scale, bh * scale
    off_x = px0 + (box_w - drawn_w) / 2 - minx * scale     # horizontally centred
    off_y = py1 - drawn_h - miny * scale                    # bottom-aligned (feet at box base)
    return [(p[0] * scale + off_x, p[1] * scale + off_y) for p in pts]


def _draw(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], dim: int) -> None:
    stick = max(2, round(dim / 200))
    joint = max(2, round(dim / 180))
    for i, (a, b) in enumerate(_LIMBS):
        (x0, y0), (x1, y1) = pts[a], pts[b]
        ang = math.atan2(y1 - y0, x1 - x0)
        dx, dy = math.sin(ang) * stick, -math.cos(ang) * stick
        draw.polygon([(x0 + dx, y0 + dy), (x1 + dx, y1 + dy), (x1 - dx, y1 - dy),
                      (x0 - dx, y0 - dy)], fill=_COLORS[i % len(_COLORS)])
    for k, (x, y) in enumerate(pts):
        draw.ellipse([x - joint, y - joint, x + joint, y + joint], fill=_COLORS[k % len(_COLORS)])


def build_skeleton(pose_spec: dict, box: list[float], width: int, height: int) -> Image.Image:
    """Render a single human pose (parametric) into ``box`` as an OpenPose control image."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    pts = _fit_to_box(_local_keypoints(pose_spec or {}), box, width, height)
    _draw(ImageDraw.Draw(img), pts, min(width, height))
    return img


def skeletons_from_plan(plan: dict, width: int, height: int) -> Image.Image | None:
    """Compose an OpenPose control image for every human entity in a scene plan.

    Returns None if the plan has no human poses (caller then skips the OpenPose control).
    """
    humans = [e for e in plan.get("entities", [])
              if isinstance(e.get("pose"), dict) and e["pose"].get("kind", "human") == "human"]
    if not humans:
        return None
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for e in humans:
        pts = _fit_to_box(_local_keypoints(e["pose"]), e["box"], width, height)
        _draw(draw, pts, min(width, height))
    return img
