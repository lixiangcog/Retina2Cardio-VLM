"""Three-task rewards transcribed from SAM-Veteran, ICLR 2026, pp. 5-6, 15-17.

No diagnostic class or teacher report enters any reward. The retinal adaptation
retains boundary pixels and only exposes the final target-specific geometry modes.
"""
import json
import math
import re

import numpy as np

SIZE = 840
TASKS = ("TG", "MC", "AUX")
TARGET_MODES = {'vessel_geometry_v1': 'retinal_vessels', 'faz_geometry_v1': 'faz'}
MODES = tuple(TARGET_MODES)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_action(response, task, size=SIZE):
    """Return action and independent format components; never repair coordinates.

    Pixel centers are integer coordinates 0..size-1. Box right/bottom may equal
    size (exclusive edge). This explicit convention is a local implementation
    choice: the paper does not release its parser or rounding implementation.
    """
    if task not in TASKS:
        raise ValueError("Unknown task")
    tags = re.fullmatch(r"\s*<think>[^<>]*</think>\s*<answer>(.*?)</answer>\s*",
                        response, flags=re.S)
    fmt = float(tags is not None)
    body = tags.group(1).strip() if tags else response.strip()
    try:
        a = json.loads(body, object_pairs_hook=_unique_object,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        if not isinstance(a, dict):
            raise ValueError("Object required")
        def coords(x, length, upper):
            return (isinstance(x, list) and len(x) == length
                    and all(type(v) is int and 0 <= v <= upper for v in x))
        if task == "TG":
            b = a.get("bbox_2d")
            if (set(a) != {"bbox_2d"} or not coords(b, 4, size)
                    or not (b[0] < b[2] and b[1] < b[3])):
                raise ValueError("Invalid box")
        else:
            if set(a) != {"pos_point", "neg_point"}:
                raise ValueError("Both point keys required, including nulls")
            if any(p is not None and not coords(p, 2, size-1) for p in a.values()):
                raise ValueError("Invalid point")
            if a['pos_point'] is not None and a['pos_point'] == a['neg_point']:
                raise ValueError("Conflicting positive/negative labels at the same pixel")
        return a, {"tags": fmt, "parseable": 1.0}
    except (ValueError, TypeError, json.JSONDecodeError):
        return None, {"tags": fmt, "parseable": 0.0}


def masks(prediction, gold):
    p, g = np.asarray(prediction), np.asarray(gold)
    if p.dtype != bool or g.dtype != bool or p.ndim != 2 or p.shape != g.shape:
        raise ValueError("Boolean same-geometry 2D masks required")
    if not g.any() or g.all():
        raise ValueError("Gold must contain a target and background")
    return p, g


def mask_iou(prediction, gold, mode):
    p, g = masks(prediction, gold)
    if mode not in MODES:
        raise ValueError("Choose an explicit reward mode")
    return float((p & g).sum()/(p | g).sum())


def box_iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter)


def gold_box(gold):
    masks(gold, gold)
    y, x = np.nonzero(gold)
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]


def case_for_mask(current, gold, mode):
    value = mask_iou(current, gold, mode)
    if mode in TARGET_MODES:
        from .target_rewards import geometry_quality
        q = geometry_quality(current, gold, TARGET_MODES[mode])
        # Prevent high IoU alone from rewarding acceptance of poor geometry.
        return 'good' if value >= .9 and q['quality'] >= .9 else 'need'
    raise ValueError('A registered target-specific geometry mode is required')


def action_case(action):
    if action is None:
        return "error"
    if "bbox_2d" in action:
        return "box"
    pos, neg = action["pos_point"] is not None, action["neg_point"] is not None
    return "both" if pos and neg else "positive" if pos else "negative" if neg else "terminate"


def delta_reward(delta):
    return 0.0 if delta <= 0 else 1.0 if delta <= .1 else 2.0 if delta <= .5 else 3.0


def _point_reward(point, region):
    if point is None or not region[point[1], point[0]]:
        return 0.0
    import cv2
    _, components = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    component = components == components[point[1], point[0]]
    y, x = np.nonzero(component)
    distance = math.hypot(point[0]-float(x.mean()), point[1]-float(y.mean()))
    return 1.0 + float(distance < 50)


def score(response, task, gold, mode, *, current=None, sam=None):
    """sam(action)->actual returned mask; GT is private reward-side data only.

    For MC the SAM callback must restore exactly the initial mask/box/logits for
    EACH rollout. A stop action bypasses SAM and keeps the actual current mask.
    """
    masks(gold, gold)
    action, fmt = parse_action(response, task, gold.shape[0])
    result = {"format_tags": fmt["tags"], "format_parseable": fmt["parseable"],
              "box_iou_reward": 0., "box_l1_reward": 0., "sam_iou": 0.,
              "decision_and_improvement": 0., "aux_decision": 0., "aux_point": 0.}
    # Validate input state EVEN if the sampled response is malformed.
    case = None
    geometry = None
    if task in ("MC", "AUX"):
        masks(current, gold)
    if task == "MC":
        case = case_for_mask(current, gold, mode)
        if case == "exclude":
            raise ValueError("MC IoU in [0.9,1) is excluded by the paper")
    if action is not None:
        if task == "TG":
            b, gt = action["bbox_2d"], gold_box(gold)
            result["box_iou_reward"] = float(box_iou(b, gt) > .5)
            result["box_l1_reward"] = float(np.abs(np.asarray(b)-gt).mean() < 10)
            predicted = sam(action)
            result["sam_iou"] = mask_iou(predicted, gold, mode)
            if mode in TARGET_MODES:
                from .target_rewards import geometry_quality
                geometry = {'after': geometry_quality(predicted, gold, TARGET_MODES[mode])}
                result['target_geometry'] = geometry['after']['quality']
        elif task == "MC":
            kind = action_case(action)
            updated = current if kind == "terminate" else sam(action)
            before, after = mask_iou(current, gold, mode), mask_iou(updated, gold, mode)
            result["sam_iou"] = after
            if mode in TARGET_MODES:
                from .target_rewards import geometry_quality
                geometry = {k: geometry_quality(v, gold, TARGET_MODES[mode])
                            for k, v in [('before', current), ('after', updated)]}
                calls = int(action['pos_point'] is not None)+int(action['neg_point'] is not None)
                result['decision_and_improvement'] = (4. if case == 'good' else 0.) if kind == 'terminate' else (
                    10*(geometry['after']['quality']-geometry['before']['quality'])-.05*calls)
        else:
            for key, region in (("pos_point", gold & ~current), ("neg_point", current & ~gold)):
                point = action[key]
                result["aux_decision"] += float((point is not None) == bool(region.any()))
                result["aux_point"] += _point_reward(point, region)
    return {"total": sum(result.values()), "components": result,
            "action_case": action_case(action), "mc_case": case, "mode": mode,
            "geometry_diagnostics": geometry}


def dynamic_select(sample, task, iteration, *, n=8, max_try=3, stop_iter=300):
    """Bounded appendix A.2 sampling. Quotas/NMS depend on actions, not rewards.

    NMS ties use sample order; the original confidence-ranking rule is not given.
    Always log fallback and retained counts. Never inject fabricated responses.
    """
    if n != 8 or max_try != 3:
        raise ValueError("Paper sampler requires 8 rollouts and 3 attempts")
    if iteration >= stop_iter:
        return sample(n), {"dynamic": False, "sampled": n, "fallback": 0}
    selected, boxes = [], []
    quotas = {"error": 2, "terminate": 1, "positive": 2, "negative": 2, "both": 1}
    sampled = 0
    for _ in range(max_try):
        candidates = sample(n)
        if len(candidates) != n:
            raise ValueError("Sampler must return the requested number")
        sampled += n
        for item in candidates:
            action, fmt = parse_action(item["text"], task)
            if not all(fmt.values()):
                action = None
            kind = action_case(action)
            if task == "TG" and kind == "box":
                accept = all(box_iou(action["bbox_2d"], b) <= .8 for b in boxes)
                if accept:
                    boxes.append(action["bbox_2d"])
            else:
                accept = quotas[kind] > 0
                if accept:
                    quotas[kind] -= 1
            if accept:
                selected.append(item)
            if len(selected) == n:
                return selected, {"dynamic": True, "sampled": sampled, "fallback": 0}
    remaining = n-len(selected)
    # Algorithm appends the first remaining entries of one more ordinary draw.
    candidates = sample(n)
    if len(candidates) != n:
        raise ValueError("Sampler returned incomplete fallback")
    selected.extend(candidates[:remaining])
    return selected, {"dynamic": True, "sampled": sampled+n, "fallback": remaining}
