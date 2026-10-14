"""Frozen CFP author tool with explicit point-directed crop rechecking.

RetSAM's public Swin model has no SAM prompt encoder. Points choose source-image
crop locations; actual author-model predictions perform localized corrections.
No pixel is added or removed merely because a point label requests it.
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image

from retina2cardio.utils import digest
from .rewards import SIZE, gold_box


def localized_update(current, crop_mask, origin, point, label, radius=48):
    """Positive adds verified crop foreground; negative removes crop background."""
    x0,y0 = origin
    h,w = crop_mask.shape
    if current.dtype != bool or crop_mask.dtype != bool or not (0<=x0<x0+w<=SIZE and 0<=y0<y0+h<=SIZE):
        raise ValueError('Invalid actual crop mask geometry')
    if label not in (0,1):
        raise ValueError('Point label must be 0 or 1')
    yy,xx = np.ogrid[y0:y0+h,x0:x0+w]
    zone = (xx-point[0])**2+(yy-point[1])**2 <= radius**2
    result = current.copy()
    part = result[y0:y0+h,x0:x0+w]
    if label:
        part[zone & crop_mask] = True
    else:
        part[zone & ~crop_mask] = False
    return result


class FrozenCFPCropTool:
    source_binding_required = True
    target = 'retinal_vessels'
    def __init__(self,runtime,cache_root,target='retinal_vessels'):
        from retina2cardio.segmentation.cfp_specialist import CFPAuthorSpecialist
        if target not in ('retinal_vessels','optic_disc'):
            raise ValueError('No verified CFP head for this target')
        self.target = target
        if target == 'optic_disc':
            from .disc_tool import CFPDiscSpecialist
            self.specialist = CFPDiscSpecialist(runtime)
        else:
            self.specialist = CFPAuthorSpecialist(runtime)
        self.specialist.model.requires_grad_(False)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True,exist_ok=False)
        self.cache = {}
        self.source = None
        self.current = self.initial_mask = self.initial_box = None
        self.provenance = {**self.specialist.provenance,
            'backend':'frozen_CFP_author_prediction_with_point_directed_crop_recheck_v1',
            'modality':'CFP','segmentation_target':target,
            'author_target_head':'od_oc union for whole disc' if target=='optic_disc' else 'artery_vein union for total vessels',
            'author_model_has_SAM_prompt_encoder':False,
            'actual_point_capability':'select original-image crop then rerun frozen author model',
            'crop_half_width_actor_pixels':120,'edit_radius_actor_pixels':48,
            'point_updates_require_actual_crop_foreground_or_background':True,
            'gold_used_to_select_crop_prediction':False}

    def bind_record(self,source):
        if (set(source) != {'image','image_sha256','modality','segmentation_target'}
                or source['modality']!='CFP' or source['segmentation_target']!=self.target):
            raise ValueError('CFP crop tool requires source-only CFP binding')
        if digest(source['image']) != source['image_sha256']:
            raise ValueError('Source image changed')
        self.source = dict(source)

    def set_image(self,image):
        if self.source is None or image.size != (SIZE,SIZE):
            raise ValueError('Bind source and provide the declared actor image geometry')
        sha = self.source['image_sha256']
        if sha not in self.cache:
            candidate,route = self.specialist.candidate({'modality':'CFP','image_path':self.source['image'],
                                                        'source_sha256':sha})
            if candidate is None:
                raise ValueError('Source specialist candidate unavailable: '+str(route))
            full = np.asarray(Image.fromarray(candidate['mask'].astype(np.uint8)*255).resize(
                (SIZE,SIZE),Image.Resampling.NEAREST))>0
            self.cache[sha] = full.copy()
        self.full_mask = self.cache[sha].copy()
        self.current = self.initial_mask = self.initial_box = None

    def initialize_box(self,box):
        x0,y0,x1,y1 = box
        if any(type(v) is not int for v in box) or not (0<=x0<x1<=SIZE and 0<=y0<y1<=SIZE):
            raise ValueError('Invalid tool box')
        self.initial_box = list(box)
        self.initial_mask = np.zeros((SIZE,SIZE),bool)
        self.initial_mask[y0:y1,x0:x1] = self.full_mask[y0:y1,x0:x1]
        self.current = self.initial_mask.copy()
        return self.current.copy()

    def prepare_mc(self,gold):
        return self.initialize_box(gold_box(gold))

    def crop_prediction(self,point):
        x,y = point
        x0,y0,x1,y1 = max(0,x-120),max(0,y-120),min(SIZE,x+121),min(SIZE,y+121)
        key = (self.source['image_sha256'],x0,y0,x1,y1)
        if key not in self.cache:
            with Image.open(self.source['image']) as im:
                original = im.convert('RGB')
            w,h = original.size
            # Actor pixels refer to the stretched paper grid. Read the matching
            # rectangle from the actual original, then invert the author's crop.
            native_box = (int(np.floor(x0*w/SIZE)),int(np.floor(y0*h/SIZE)),
                          int(np.ceil(x1*w/SIZE)),int(np.ceil(y1*h/SIZE)))
            path = self.cache_root/(key[0]+f'_{x0}_{y0}_{x1}_{y1}.png')
            original.crop(native_box).save(path)
            candidate,route = self.specialist.candidate({'modality':'CFP','image_path':str(path),
                                                        'source_sha256':digest(path)})
            if candidate is None:
                # Unsupported local crop is reported, rather than a fabricated mask.
                self.cache[key] = None
            else:
                self.cache[key] = np.asarray(Image.fromarray(candidate['mask'].astype(np.uint8)*255).resize(
                    (x1-x0,y1-y0),Image.Resampling.NEAREST))>0
                mask_path = path.with_name(path.stem+'_prediction.png')
                Image.fromarray(self.cache[key].astype(np.uint8)*255).save(mask_path)
                path.with_suffix('.json').write_text(json.dumps({
                    'original_source_sha256':self.source['image_sha256'],
                    'original_pixel_box_xyxy':native_box,'actor_pixel_box_xyxy':[x0,y0,x1,y1],
                    'crop_image_sha256':digest(path),'crop_mask_sha256':digest(mask_path),
                    'actual_author_model_provenance':candidate['provenance'],
                    'annotation_mask_used':False},indent=2))
        return self.cache[key],(x0,y0)

    def refine(self,action,base):
        pos,neg = action['pos_point'],action['neg_point']
        if pos is not None and pos==neg:
            raise ValueError('Conflicting point labels')
        current = base.copy()
        for point,label in ((pos,1),(neg,0)):
            if point is not None:
                crop,origin = self.crop_prediction(point)
                if crop is not None:
                    current = localized_update(current,crop,origin,point,label)
        return current

    def __call__(self,action):
        if 'bbox_2d' in action:
            # Training TG rewards must not mutate the MC initial state.
            box = action['bbox_2d']
            result = np.zeros_like(self.full_mask)
            x0,y0,x1,y1 = box
            result[y0:y1,x0:x1] = self.full_mask[y0:y1,x0:x1]
            return result
        if self.initial_mask is None:
            raise RuntimeError('MC needs an actual initial specialist state')
        return self.refine(action,self.initial_mask)

    def step(self,action):
        if self.current is None:
            raise RuntimeError('Initialize the box before refinement')
        self.current = self.refine(action,self.current)
        return self.current.copy()
