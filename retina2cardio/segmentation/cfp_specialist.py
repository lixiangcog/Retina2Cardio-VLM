"""Strict adapter to unchanged author RetSAM model/preprocessing, not Meta SAM.

Only public predicted vascular masks are produced. Training-package initializers,
unsafe checkpoint fallback and permissive partial weight loading are not used.
"""
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
from PIL import Image
from retina2cardio.utils import digest, read, technical_gate

COMMIT = 'a78d16324ea3f98e95de3bfd34aab0527e3bcfb9'
WEIGHT_SHA = '3d309abd8fbc59adef7473f920506234d8d3446d68b8f7905eeaf1680b872ae4'


def restore_crop(mask, crop, source_wh):
    """Invert the real crop/resize onto the full source canvas, never stretch it."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Binary model-grid mask required')
    w, h = source_wh
    x0, x1, y0, y1 = (crop[k] for k in ('xmin','xmax','ymin','ymax'))
    if not all(type(v) in (int,np.int32,np.int64) for v in (x0,x1,y0,y1)) or not (0<=x0<x1<=w and 0<=y0<y1<=h):
        raise ValueError('Invalid actual crop bounds')
    result = np.zeros((h,w), dtype=bool)
    small = Image.fromarray(mask.astype(np.uint8)*255)
    result[y0:y1,x0:x1] = np.asarray(small.resize((x1-x0,y1-y0),Image.Resampling.NEAREST)) > 0
    return result


def verified_sources(runtime):
    runtime = Path(runtime).resolve()
    manifest = read(runtime/'source_manifest.json')
    if manifest['commits'].get('Wzhjerry__RetSAM') != COMMIT:
        raise ValueError('RetSAM source revision is not pinned')
    expected = []
    for row in manifest['files']:
        archive = row['archive'].replace('\\', '/')
        relative = Path(archive)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe author source manifest path')
        if archive.startswith('Wzhjerry__RetSAM/'):
            expected.append({**row, 'archive': archive})
    if not expected:
        raise ValueError('Missing author source hashes')
    for row in expected:
        path = (runtime/row['archive']).resolve()
        if not path.is_relative_to(runtime) or digest(path) != row['sha256']:
            raise ValueError('Author source or license changed')
    return runtime/'Wzhjerry__RetSAM'


def source_module(root, relative):
    """Import exact source under an isolated name, avoiding training-only init."""
    prefix = '_verified_retsam_adapter'
    for name, path in ((prefix,root),(prefix+'.models',root/'models'),
                       (prefix+'.inference_modules',root/'inference_modules')):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        elif list(sys.modules[name].__path__) != [str(path)]:
            raise ValueError('Cannot mix author source directories in one process')
    name = prefix+'.'+relative.replace('/','.').removesuffix('.py')
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name,root/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_state(state):
    if not isinstance(state,dict) or not state:
        raise ValueError('Missing actual author state dictionary')
    result = {}
    for key,value in state.items():
        name = key
        while name.startswith(('module.','model.')):
            name = name.split('.',1)[1]
        if name in result:
            raise ValueError('Colliding weight prefixes')
        result[name] = value
    return result


class CFPAuthorSpecialist:
    def __init__(self,runtime,device='cuda'):
        import torch
        runtime = Path(runtime).resolve()
        ready = read(runtime/'ready_from_local_transfer.json')
        checkpoint = Path(ready['checkpoint'])
        if ready.get('checkpoint_sha256') != WEIGHT_SHA or digest(checkpoint) != WEIGHT_SHA:
            raise ValueError('Author public CFP weight checksum differs')
        source = verified_sources(runtime)
        config = source_module(source,'inference_modules/config.py').InferenceConfig()
        self.processor = source_module(source,'inference_modules/image_processor.py').ImageProcessor()
        self.noise_filter = source_module(source,'inference_modules/postprocessing.py').NoiseFilter(config)
        self.mask_processor = source_module(source,'inference_modules/utils.py').MaskProcessor
        # Safe data-only deserialization; do not copy author's unsafe pickle fallback.
        bundle = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        state = clean_state(bundle.get('state_dict',bundle))
        params = dict(config.model_params,out_channels=config.default_output_channels)
        has_coordinate = any(k.startswith('coordinate_head.') for k in state)
        if has_coordinate:
            coordinate = state['coordinate_head.coordinate_regressor.6.weight']
            module = source_module(source,'models/swin_multitask_final.py')
            self.model = module.Swin_Multitask_Final(**params,enable_coordinate_prediction=True,
                                                    num_coordinates=coordinate.shape[0])
        else:
            module = source_module(source,'models/swin_multitask.py')
            self.model = module.Swin_Multitask(**params)
        self.model.load_state_dict(state,strict=True)
        self.model = self.model.to(device).eval()
        self.threshold = config.get_prob_threshold('artery_vein')
        self.device, self.torch = device,torch
        self.provenance = {'backend':'author_RetSAM_public_SwinUNETR_CFP_specialist',
            'adapter_source_sha256':digest(__file__),
            'source_repository':'https://github.com/Wzhjerry/RetSAM','source_commit':COMMIT,
            'source_manifest_sha256':digest(runtime/'source_manifest.json'),
            'checkpoint_sha256':WEIGHT_SHA,'weights_strictly_loaded':True,
            'checkpoint_safe_weights_only_load':True,'original_author_preprocessing_reused':True,
            'input_grid_hw':[640,640],'artery_vein_probability_gate':self.threshold,
            'artery_vein_final_binary_threshold':0.5,
            'author_vessel_component_noise_filter_reused':True,
            'minimum_component_pixels_in_model_grid':config.get_analysis_min_component_size('artery_vein'),
            'threshold_is_author_default_not_patient_test_tuned':True,
            'model_is_Meta_SAM':False,'point_box_mask_prompt_refinement_supported':False,
            'class_identity_is_unvalidated_model_prediction':True,
            'clinical_measurements_released':False,'clinical_domain_validation':False,
            'adaptations':['isolated import avoids training-only package initializers',
                           'strict safe weight load instead of permissive source loader',
                           'restore actual crop on source canvas for agent feedback',
                           'author AV noise filter and saved-mask decode; no unreviewed clinical analysis']}

    def candidate(self,job):
        if job['modality'] != 'CFP':
            raise ValueError('CFP tool cannot operate on OCTA/OCT')
        if digest(job['image_path']) != job['source_sha256']:
            raise ValueError('Frozen CFP source changed')
        prepared = self.processor.load_and_preprocess(job['image_path'])
        with self.torch.inference_mode():
            outputs = self.model(prepared['tensor'].to(self.device))
            heads = outputs[0] if isinstance(outputs,tuple) else outputs
            logits = heads[0]
            if tuple(logits.shape) != (1,2,640,640) or not self.torch.isfinite(logits).all():
                raise ValueError('Expected actual public two-channel AV head')
            filtered = self.noise_filter.filter_prediction(logits,'artery_vein')
            _,binary = self.mask_processor.process_multilabel_prediction(filtered)
            binary = binary.astype(bool)
        native = np.logical_or(binary[0],binary[1])
        full = restore_crop(native,prepared['crop_coords'],prepared['original_shape'])
        gate = technical_gate(full)
        if gate:
            return None, {'tool':'abstain','reason':'degenerate_CFP_specialist_'+gate}
        provenance = {**self.provenance,'source_sha256':job['source_sha256'],
            'crop_xyxy':{k:int(v) for k,v in prepared['crop_coords'].items()},
            'source_grid_wh':list(prepared['original_shape']),
            'source_geometry':'inverse nearest resize into actual crop on zero full-image canvas'}
        return {'mask':full,'native_mask':native,'provenance':provenance}, {
            'tool':'author_RetSAM_CFP_vessel_candidate','SAM_refinement_enabled':False,
            'measurement_release_approved':False}
