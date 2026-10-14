import hashlib,json
from pathlib import Path
import numpy as np

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def technical_gate(mask):
    # Necessary, not sufficient: these checks never establish anatomical accuracy.
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Full-resolution binary mask required')
    if not mask.any():
        return 'empty_mask'
    if mask.all():
        return 'all_foreground_mask'
    return None
