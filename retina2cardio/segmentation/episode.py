"""Source-only retinal-vessel agent execution; no annotation or diagnosis input."""
import hashlib
import json

import numpy as np
from PIL import Image

from retina2cardio.utils import digest
from .rewards import SIZE,action_case,parse_action


def vessel_mask_gate(mask):
    """Necessary technical check only; does not establish anatomical accuracy."""
    if not mask.any():
        return "empty_vessel_mask"
    if float(mask.mean()) > .45:
        return "dense_region_not_vessel_network"
    return None


def run_steps(actor, sam, record, image, mask_trace=None):
    from .runtime import bind_image
    from .region_evidence import region_gate
    target = record.get('segmentation_target', 'retinal_vessels')
    if target not in ('retinal_vessels','optic_disc','faz'):
        raise ValueError('Unknown segmentation target')
    gate_fn = vessel_mask_gate if target == 'retinal_vessels' else region_gate
    bind_image(sam,record,image)
    actions = []
    seen = set()
    unchanged = 0
    refinements = 0
    mask = np.zeros((image.height,image.width),dtype=bool)
    # Initial box + three refinements + one terminal decision. The terminal
    # observation cannot trigger a fourth tool call, but may accept the result.
    for step in range(5):
        task = "TG" if step == 0 else "MC"
        inputs = actor.inputs(image, task, record["query"], mask if step else None)
        actor.model.eval()
        with actor.torch.inference_mode():
            tokens = actor.model.generate(**inputs, do_sample=False, max_new_tokens=actor.max_response,
                                          repetition_penalty=1.1, use_cache=True)
        text = actor.processor.tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        action, fmt = parse_action(text, task)
        actions.append({"step": step, "response": text, "format": fmt, "action": action})
        if action is None or fmt['tags'] != 1.:
            return mask, actions, "invalid_action"
        if step == 0:
            # NEVER sam.prepare_mc(gold) here. Evaluation starts from actor box.
            mask = sam.initialize_box(action['bbox_2d'])
        elif action_case(action) == "terminate":
            gate = gate_fn(mask)
            return mask, actions, "acceptance_blocked_"+gate if gate else "accepted"
        else:
            if refinements >= 3:
                return mask,actions,'refinement_budget_exhausted_not_accepted'
            key = (hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(), json.dumps(action, sort_keys=True))
            if key in seen:
                return mask, actions, "repeated_action_not_accepted"
            seen.add(key)
            previous = mask.copy()
            # Backend exposes its actual capability: SAM2 mask-logit feedback or
            # point-directed author CFP crop rechecking. No gold-guided selection.
            mask = sam.step(action)
            refinements += 1
            unchanged = unchanged+1 if np.array_equal(previous, mask) else 0
        actions[-1]['returned_foreground_fraction'] = float(mask.mean())
        actions[-1]['technical_gate'] = gate_fn(mask)
        actions[-1]['refinements_used'] = refinements
        if mask_trace is not None:
            mask_trace.append(mask.copy())
        if unchanged >= 2:
            return mask, actions, "unchanged_mask_not_accepted"
    return mask, actions, "max_refinement_steps"


def run_source_episode(actor,tool,source,mask_trace=None):
    required = {"image","image_sha256","modality","query","segmentation_target"}
    if set(source)!=required or source["modality"]!="CFP" or source["segmentation_target"]!="retinal_vessels":
        raise ValueError("This source-only entry point supports CFP retinal vessels only")
    if digest(source["image"])!=source["image_sha256"]:
        raise ValueError("Source image changed")
    with Image.open(source["image"]) as original:
        image=original.convert("RGB").resize((SIZE,SIZE),Image.Resampling.BICUBIC)
    return run_steps(actor,tool,source,image,mask_trace)


def run_octa_source_episode(actor,tool,source,mask_trace=None):
    required={'image','image_sha256','modality','query','segmentation_target'}
    if (set(source)!=required or source['modality']!='OCTA' or source['segmentation_target']!='retinal_vessels'
            or tool.provenance.get('modality')!='OCTA'
            or tool.provenance.get('native_point_box_mask_prompt_supported') is not True):
        raise ValueError('Native OCTA vessel feedback needs the correct verified target tool')
    if digest(source['image'])!=source['image_sha256']:raise ValueError('Original OCTA source changed')
    with Image.open(source['image']) as original:
        image=original.convert('RGB').resize((SIZE,SIZE),Image.Resampling.BICUBIC)
    return run_steps(actor,tool,source,image,mask_trace)


def run_region_source_episode(actor,tool,source,mask_trace=None):
    """Disc/FAZ feedback with exact tool routing; no annotation in this entry."""
    from .region_evidence import TARGET_MODALITY
    required={'image','image_sha256','modality','query','segmentation_target'}
    target=source.get('segmentation_target')
    if (set(source)!=required or TARGET_MODALITY.get(target)!=source['modality']
            or tool.provenance.get('segmentation_target')!=target
            or tool.provenance.get('modality')!=source['modality']):
        raise ValueError('Dedicated region tool and source-only target binding required')
    if digest(source['image'])!=source['image_sha256']:raise ValueError('Region source changed')
    with Image.open(source['image']) as original:
        image=original.convert('RGB').resize((SIZE,SIZE),Image.Resampling.BICUBIC)
    return run_steps(actor,tool,source,image,mask_trace)
