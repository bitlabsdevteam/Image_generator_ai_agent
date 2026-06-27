"""Scene planning: turn ANY prompt into a structured layout the generator can enforce.

CLIP-guided SDXL has no spatial operator, so "girl standing on top of a hamburger" collapses
to "girl + hamburger" with the strongest noun prior winning. The fix is to stop relying on the
text encoder for composition: a local LLM parses the request into a structured *scene plan*
(entities, attributes, spatial relations, normalized boxes, and a parametric pose per human),
which downstream stages turn into ControlNet conditioning and into the checklist a vision model
verifies.

Everything here is prompt-derived at runtime — no pose presets, no keyword tables. The only
fixed pieces are (a) the JSON schema the LLM fills and (b) ``normalize_layout``, a general
predicate->geometry resolver that enforces whatever relations the LLM extracted. Both generalize
to any subject/relation.

Degrades gracefully (mirrors ``llm_prompt._ollama_expand`` / ``interview.generate_questions``):
on any failure ``plan_scene`` returns a single full-frame entity so generation still works.
"""
from __future__ import annotations

import sys

from .config import CONFIG
from ._jsonio import extract_json

# Pose joint-angle convention (degrees), documented for the LLM and consumed by the forward
# kinematics in ``pose_control``. Shoulder/hip angles are measured from the downward torso axis:
# 0 = limb hangs straight down, 90 = straight out sideways, 180 = straight up overhead. Elbow/
# knee are flex angles: 0 = straight. All optional — missing joints fall back to a neutral rig.
_POSE_HINT = (
    "pose.joints uses degrees: shoulder/hip 0=limb down, 90=out sideways, 180=up overhead; "
    "elbow/knee 0=straight, 90=bent. e.g. arms raised overhead => l_shoulder_deg/r_shoulder_deg "
    "~170, elbows ~10."
)

_SYS = (
    "/no_think\n"
    "You are a scene-layout director for an image generator. Convert the user's request into a "
    "STRICT JSON scene plan and output JSON ONLY (no prose, no code fences). Schema:\n"
    '{"entities":[{"id":"<snake_id>","kind":"human|object",'
    '"phrase":"<concise visual description, keep named colours/brands, no style words>",'
    '"box":[x0,y0,x1,y1],'
    '"pose":null|{"stance":"standing|sitting|crouching|lying",'
    '"joints":{"l_shoulder_deg":..,"r_shoulder_deg":..,"l_elbow_deg":..,"r_elbow_deg":..,'
    '"l_hip_deg":..,"r_hip_deg":..,"l_knee_deg":..,"r_knee_deg":..}}}],'
    '"relations":[{"subject":"<id>","predicate":"<on top of|inside|behind|in front of|above|'
    'below|left of|right of|holding>","object":"<id>"}],'
    '"global":{"palette":"<colours/brand>","mood":"<mood>"}}\n'
    "Rules: boxes normalized [0,1], origin top-left, x0<x1 and y0<y1; place entities to satisfy "
    "the relations (e.g. 'A on top of B' => A's box ABOVE and resting on B's box). kind:'human' "
    "is ONLY for people / humanoid characters and they MUST have a non-null pose matching the "
    "requested action; animals, creatures and things are kind:'object' with pose:null. "
    + _POSE_HINT + "\n"
    "For STANDING use straight legs (hip ~8, knee ~5); only bend hips/knees (~90) for SITTING or "
    "crouching.\n"
    'Example for "a robot standing on a platform, both arms raised in victory":\n'
    '{"entities":[{"id":"robot","kind":"human","phrase":"victory robot","box":[0.36,0.05,0.64,0.62],'
    '"pose":{"stance":"standing","joints":{"l_shoulder_deg":168,"r_shoulder_deg":168,'
    '"l_elbow_deg":10,"r_elbow_deg":10,"l_hip_deg":8,"r_hip_deg":8,"l_knee_deg":5,'
    '"r_knee_deg":5}}},{"id":"platform","kind":"object","phrase":"raised platform","box":[0.2,0.62,0.8,0.95],'
    '"pose":null}],"relations":[{"subject":"robot","predicate":"on top of","object":"platform"}],'
    '"global":{"palette":"steel blue","mood":"triumphant"}}'
)

_PREDICATES = {
    "on top of", "on", "standing on", "inside", "in", "within", "behind",
    "in front of", "above", "below", "under", "left of", "right of", "holding", "next to",
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _valid_box(box) -> list[float] | None:
    """Return a clamped [x0,y0,x1,y1] with positive area, or None if unusable."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (_clamp01(v) for v in box)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _fallback_plan(user_request: str) -> dict:
    """Single full-frame entity — generation behaves exactly like plain text-to-image."""
    return {
        "entities": [{"id": "scene", "phrase": user_request.strip(), "box": [0.0, 0.0, 1.0, 1.0],
                      "pose": None}],
        "relations": [],
        "global": {},
        "fallback": True,
    }


def _validate(plan: dict, user_request: str) -> dict:
    """Coerce a raw LLM plan into a well-formed plan; fall back if nothing usable survives."""
    raw_entities = plan.get("entities") if isinstance(plan, dict) else None
    if not isinstance(raw_entities, list):
        return _fallback_plan(user_request)

    entities: list[dict] = []
    for i, e in enumerate(raw_entities):
        if not isinstance(e, dict):
            continue
        phrase = str(e.get("phrase", "")).strip()
        box = _valid_box(e.get("box"))
        if not phrase or box is None:
            continue
        eid = str(e.get("id") or f"e{i}").strip()
        pose = e.get("pose") if isinstance(e.get("pose"), dict) else None
        kind = str(e.get("kind", "")).strip().lower()
        if kind not in ("human", "object"):
            kind = "human" if pose is not None else "object"
        # A human must have a body to pose: if the model left it null, give a neutral standing
        # rig so placement/stance are still enforced (the verifier loop refines the joints).
        if kind == "human" and pose is None:
            pose = {"stance": "standing", "joints": {}}
        if pose is not None:
            pose.setdefault("kind", "human")
        entities.append({"id": eid, "kind": kind, "phrase": phrase, "box": box, "pose": pose})

    if not entities:
        return _fallback_plan(user_request)

    ids = {e["id"] for e in entities}
    relations = []
    for r in (plan.get("relations") or []):
        if not isinstance(r, dict):
            continue
        subj, obj = str(r.get("subject", "")), str(r.get("object", ""))
        pred = str(r.get("predicate", "")).strip().lower()
        if subj in ids and obj in ids and subj != obj and pred in _PREDICATES:
            relations.append({"subject": subj, "predicate": pred, "object": obj})

    glob = plan.get("global") if isinstance(plan.get("global"), dict) else {}
    return {"entities": entities, "relations": relations, "global": glob, "fallback": False}


def _apply_relation(subj: list[float], obj: list[float], predicate: str) -> list[float]:
    """Reposition the subject box so it satisfies ``predicate`` relative to the object box.

    Preserves the subject's size and moves it; a general geometric resolver that works for any
    pair of entities. Unhandled predicates leave the subject untouched.
    """
    sw, sh = subj[2] - subj[0], subj[3] - subj[1]
    ocx = (obj[0] + obj[2]) / 2
    p = predicate

    def centered_x(width: float) -> tuple[float, float]:
        x0 = _clamp01(ocx - width / 2)
        return x0, min(1.0, x0 + width)

    if p in ("on top of", "on", "standing on", "above"):
        x0, x1 = centered_x(sw)
        y1 = obj[1]              # subject's feet rest on the object's top edge
        y0 = max(0.0, y1 - sh)
        return [x0, y0, x1, y1]
    if p in ("below", "under"):
        x0, x1 = centered_x(sw)
        y0 = obj[3]
        return [x0, y0, x1, min(1.0, y0 + sh)]
    if p in ("inside", "in", "within"):
        # Shrink subject to sit within the object with a small margin.
        mx, my = (obj[2] - obj[0]) * 0.15, (obj[3] - obj[1]) * 0.15
        return [obj[0] + mx, obj[1] + my, obj[2] - mx, obj[3] - my]
    if p == "left of":
        x1 = obj[0]
        return [max(0.0, x1 - sw), subj[1], x1, subj[3]]
    if p == "right of":
        x0 = obj[2]
        return [x0, subj[1], min(1.0, x0 + sw), subj[3]]
    # behind / in front of / holding / next to: keep the LLM's box (overlap is intentional).
    return subj


def normalize_layout(plan: dict) -> dict:
    """Enforce each extracted relation geometrically by repositioning subject boxes.

    The relations are the user's stated intent, so we make the boxes obey them rather than
    trusting the LLM's raw coordinates. General over any entities/predicates.
    """
    by_id = {e["id"]: e for e in plan["entities"]}
    for r in plan["relations"]:
        subj_e, obj_e = by_id.get(r["subject"]), by_id.get(r["object"])
        if subj_e and obj_e:
            subj_e["box"] = _apply_relation(subj_e["box"], obj_e["box"], r["predicate"])
    return plan


def _arms_phrase(pose: dict) -> str:
    """Short natural-language hint for a pose's arms, derived from shoulder angles."""
    j = pose.get("joints") if isinstance(pose.get("joints"), dict) else {}
    try:
        l, r = float(j.get("l_shoulder_deg", 0)), float(j.get("r_shoulder_deg", 0))
    except (TypeError, ValueError):
        l = r = 0.0
    if l >= 120 and r >= 120:
        return "both arms raised overhead"
    if l >= 120 or r >= 120:
        return "one arm raised"
    return ""


def constraints_from_plan(plan: dict) -> list[dict]:
    """Derive the yes/no checklist the verifier grades, straight from the plan.

    Each constraint is a natural-language question about a spatial relation or a human pose —
    exactly the things CLIP scoring is blind to. Generated from the parse, so it covers any
    request without bespoke checks.
    """
    by_id = {e["id"]: e for e in plan.get("entities", [])}
    items: list[dict] = []
    for i, r in enumerate(plan.get("relations", [])):
        s, o = by_id.get(r["subject"]), by_id.get(r["object"])
        if s and o:
            items.append({"id": f"rel{i}",
                          "text": f"Is the {s['phrase']} {r['predicate']} the {o['phrase']}?"})
    for e in plan.get("entities", []):
        pose = e.get("pose")
        if isinstance(pose, dict):
            stance = str(pose.get("stance", "standing"))
            arms = _arms_phrase(pose)
            desc = f"{stance}" + (f" with {arms}" if arms else "")
            items.append({"id": f"pose_{e['id']}",
                          "text": f"Is the {e['phrase']} {desc}?"})
    return items


def _ollama_json(system: str, user: str) -> dict | None:
    """Call the LLM for a JSON object. Disables reasoning-model "thinking" (Qwen3 etc. otherwise
    spend minutes on <think> chains for this structured task) and uses a low temperature plus a
    token cap so structured output is fast and deterministic."""
    try:
        import ollama

        # '/no_think' steers Qwen3-family models to skip the thinking phase without depending on
        # a particular ollama-python version supporting the `think=` kwarg.
        sys_prompt = f"/no_think\n{system}"
        opts = {"temperature": 0.2, "top_p": 0.9, "num_predict": 900}
        kwargs = dict(
            model=CONFIG["llm"]["model"],
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
            options=opts,
        )
        try:
            resp = ollama.chat(think=False, **kwargs)   # newer ollama-python
        except TypeError:
            resp = ollama.chat(**kwargs)                 # older client without `think`
        return extract_json(resp["message"]["content"])
    except Exception as exc:  # ollama down, model missing, bad output, etc.
        print(f"[planner] fallback ({exc})", file=sys.stderr, flush=True)
        return None


def plan_scene(user_request: str, width: int = 1, height: int = 1) -> dict:
    """Return a validated, geometry-normalized scene plan for ``user_request``.

    ``width``/``height`` are accepted for API symmetry (boxes are resolution-independent and
    normalized); callers scale boxes to pixels when building control images.
    """
    raw = _ollama_json(_SYS, f"Request: {user_request}")
    plan = _validate(raw, user_request) if raw else _fallback_plan(user_request)
    return normalize_layout(plan)


def replan(plan: dict, failed_constraints: list[dict], user_request: str) -> dict:
    """Ask the LLM to adjust the plan to fix the verifier's failed constraints.

    Structure-level refinement (boxes/pose), not prompt rewriting — the latter cannot fix
    spatial errors. Falls back to the (re-normalized) current plan on any failure.
    """
    reasons = "; ".join(
        f"{c.get('id', '?')}: {c.get('reason', 'failed')}" for c in failed_constraints
    ) or "the composition did not match the request"
    import json as _json

    user = (
        f"Original request: {user_request}\n"
        f"Current scene plan JSON: {_json.dumps({k: plan[k] for k in ('entities','relations','global')})}\n"
        f"A vision check FAILED these constraints: {reasons}\n"
        "Return a corrected scene plan in the SAME JSON schema that fixes them — adjust boxes, "
        "poses and stances (e.g. make a figure clearly STANDING with feet at the box bottom, or "
        "move a subject's box to truly sit on top of its object). JSON only."
    )
    raw = _ollama_json(_SYS, user)
    if not raw:
        return normalize_layout(plan)
    return normalize_layout(_validate(raw, user_request))
