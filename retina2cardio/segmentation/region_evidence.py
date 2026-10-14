import numpy as np

def region_gate(mask):
    if mask.dtype != bool or mask.ndim != 2:
        raise ValueError('Boolean two-dimensional region mask required')
    if not mask.any():
        return 'empty_region'
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return 'region_touches_image_border'
    from scipy.ndimage import label
    if label(mask, np.ones((3, 3), dtype=int))[1] != 1:
        return 'disconnected_region'
    return None
