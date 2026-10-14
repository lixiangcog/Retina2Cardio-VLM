import numpy as np

def skeletonize(mask):
    """Zhang-Suen thinning; retain geometry, no downsampling for narrow vessels."""
    value = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max(mask.shape) * 2):
        changed = False
        for second in (False, True):
            p = value[1:-1, 1:-1]
            n = [value[:-2, 1:-1], value[:-2, 2:], value[1:-1, 2:], value[2:, 2:],
                 value[2:, 1:-1], value[2:, :-2], value[1:-1, :-2], value[:-2, :-2]]
            count = sum(n)
            transitions = sum(((n[i] == 0) & (n[(i+1) % 8] == 1)).astype(np.uint8) for i in range(8))
            product1 = n[0]*n[2]*(n[6] if second else n[4])
            product2 = n[0]*n[4]*n[6] if second else n[2]*n[4]*n[6]
            remove = (p == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & (product1 == 0) & (product2 == 0)
            if remove.any():
                p[remove] = 0
                changed = True
        if not changed:
            return value[1:-1, 1:-1].astype(bool)
    raise RuntimeError('Thinning failed to converge; do not emit proxy widths')
