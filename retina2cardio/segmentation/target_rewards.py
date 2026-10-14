"""Research-only geometric reward proxies; no physical calibration or CAD labels."""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, label
from retina2cardio.segmentation.metrics import skeletonize


def geometry_quality(prediction, gold, target):
    p, g = np.asarray(prediction), np.asarray(gold)
    if p.dtype != bool or g.dtype != bool or p.ndim != 2 or p.shape != g.shape:
        raise ValueError('Boolean same-geometry masks required')
    if not g.any() or g.all():
        raise ValueError('Nondegenerate independent gold required')
    if target not in ('retinal_vessels', 'faz'):
        raise ValueError('Unsupported reward target')
    iou = float((p & g).sum() / (p | g).sum())
    if target == 'retinal_vessels':
        sp, sg = skeletonize(p), skeletonize(g)
        precision = float((sp & g).sum()/max(1, sp.sum()))
        recall = float((sg & p).sum()/sg.sum())
        cldice = 2*precision*recall/max(precision+recall, 1e-12)
        # Component-count agreement is a topology proxy, not proof of vessel connectivity.
        nparts = label(p, np.ones((3, 3)))[1]
        gparts = label(g, np.ones((3, 3)))[1]
        topology = float(np.exp(-abs(nparts-gparts)/max(1, gparts))) if p.any() else 0.
        # Sample both distance fields on fixed GOLD centerline locations; missing
        # vessels retain zero predicted radius and cannot escape the denominator.
        dg = distance_transform_edt(g)[sg]
        dp = distance_transform_edt(p)[sg]
        error = float(np.mean(np.abs(dp-dg)/np.maximum(dg, 1.)))
        diameter = float(np.exp(-error)) if p.any() else 0.
        parts = dict(iou=iou, centerline_cldice=cldice,
                     component_agreement=topology, diameter_quality=diameter,
                     diameter_relative_error=error)
        quality = .4*iou + .25*cldice + .15*topology + .2*diameter
    else:
        area_error = float(abs(int(p.sum())-int(g.sum()))/g.sum())
        bp, bg = p & ~binary_erosion(p), g & ~binary_erosion(g)
        # Symmetric boundary distance relative to gold equivalent radius.
        radius = np.sqrt(g.sum()/np.pi)
        boundary_error = float((distance_transform_edt(~bg)[bp].mean()+
                                distance_transform_edt(~bp)[bg].mean())/(2*radius)) if bp.any() else None
        boundary = float(np.exp(-boundary_error)) if boundary_error is not None else 0.
        area = float(np.exp(-area_error)) if p.any() else 0.
        parts = dict(iou=iou, area_quality=area, area_relative_error=area_error,
                     boundary_quality=boundary, boundary_relative_error=boundary_error)
        quality = .5*iou + .25*area + .25*boundary
    return dict(quality=float(quality), components=parts, target=target,
                units='dimensionless_native_or_training_grid_proxy', clinically_validated=False)
