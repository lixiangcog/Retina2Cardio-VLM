IMAGE_TOKEN_INDEX=-200

def encode_text(tokenizer, text):
    pieces = text.split("<image>")
    ids = []
    for index, piece in enumerate(pieces):
        if index:
            ids.append(IMAGE_TOKEN_INDEX)
        ids.extend(tokenizer.encode(piece, add_special_tokens=False))
    return ids

def core_loading_guard(info):
    critical = ("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head.")
    missing = [k for k in info.get("missing_keys", []) if k.startswith(critical)]
    if missing or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Pretrained language weight mismatch: {info}")



def binary_ids(tokenizer):
    ids = [tokenizer.encode(str(y), add_special_tokens=False) for y in (0, 1)]
    if any(len(v) != 1 for v in ids) or ids[0] == ids[1]:
        raise ValueError('The actual tokenizer must have distinct single-token 0/1 codes')
    return [v[0] for v in ids]

from pathlib import Path
from PIL import Image
from .utils import digest

def image_inputs(views, root, model, verify=True):
    root = Path(root).resolve()
    if any('role' in v for v in views):
        roles = [v.get('role') for v in views]
        order = ['CFP', 'OCTA_UNKNOWN'] if 'OCTA_UNKNOWN' in roles else ['CFP', 'DCP', 'SCP', 'RPCP']
        if 'OCTA_UNKNOWN' in roles and roles != order:
            raise ValueError('Unknown-layer OCTA must be an independent CFP + single OCTA sample')
        if len(roles) != len(set(roles)) or roles != [r for r in order if r in roles]:
            raise ValueError('Layered image order must be CFP, DCP, SCP, RPCP without repeated roles')
        if any(v['modality'] != ('CFP' if v['role'] == 'CFP' else 'OCTA') for v in views):
            raise ValueError('Layered role/modality mismatch')
    result = []
    for modality, tower in (('CFP', model.get_vision_tower()), ('OCTA', model.get_oct_2d_tower())):
        images = []
        for view in views:
            if view['modality'] != modality:
                continue
            path = (root / view['path']).resolve()
            if not path.is_relative_to(root) or (verify and digest(path) != view['sha256']):
                raise ValueError('Source checksum/path mismatch')
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        result.append(tower.image_processor.preprocess(images, return_tensors='pt')['pixel_values'] if images else None)
    return result
