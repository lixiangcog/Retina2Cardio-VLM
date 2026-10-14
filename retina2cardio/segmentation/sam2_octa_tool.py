"""Native point/mask-feedback SAM2, specialized on public manual OCTA vessels.

Only a verified trained OCTA adapter can enter the agent route. A generic SAM2
checkpoint is still not an approved retinal segmentation tool. No gold is read
by this inference class. Vessel and FAZ heads require separate target receipts;
neither is routed to CFP or artery/vein classification.
"""
import json
from pathlib import Path
import numpy as np

from retina2cardio.utils import digest
from .runtime import FrozenSAM
from .rewards import SIZE

ARTIFACT='public_manual_OCTA_vessel_SAM2_decoder_v1'
FAZ_ARTIFACT='public_manual_OCTA_FAZ_SAM2_decoder_v1'


def source_hashes(root):
    root=Path(root).resolve()
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p.suffix in ('.py','.yaml')}


def verify_target_receipt(root,foundation,target='retinal_vessels'):
    root=Path(root).resolve()
    receipt=json.loads((root/'completion.json').read_text())
    run=json.loads((root/'run.json').read_text())
    checkpoint=root/'final_target_tool.pt'
    if target not in ('retinal_vessels','faz'):raise ValueError('Unknown OCTA target')
    artifact = ARTIFACT if target=='retinal_vessels' else FAZ_ARTIFACT
    if (receipt.get('artifact')!=artifact or receipt.get('complete') is not True
            or receipt.get('changed_parameter_tensors',0)<=0
            or receipt.get('clinical_release') is not False
            or receipt.get('checkpoint_sha256')!=digest(checkpoint)
            or run.get('modality')!='OCTA' or run.get('segmentation_target')!=target
            or run.get('manual_training_sources',0)<1
            or run.get('technical_test_only') is not False or run.get('pseudo_masks_used') is not False
            or run.get('clinical_data_loaded') is not False
            or run.get('foundation_sha256')!=digest(foundation)):
        raise ValueError('A verified public-manual OCTA-specific SAM2 adapter is required')
    import sam2
    if not run.get('sam2_source_sha256') or source_hashes(Path(sam2.__file__).parent)!=run['sam2_source_sha256']:
        raise ValueError('SAM2 source/configuration changed since OCTA tool training')
    return checkpoint,receipt,run


class FrozenOCTASAM2(FrozenSAM):
    source_binding_required=True
    target='retinal_vessels'

    def __init__(self,foundation,runtime,target='retinal_vessels'):
        self.target=target
        path,receipt,run=verify_target_receipt(runtime,foundation,target)
        artifact = ARTIFACT if target=='retinal_vessels' else FAZ_ARTIFACT
        super().__init__(foundation)
        bundle=self.torch.load(path,map_location='cpu',weights_only=True)
        if bundle['artifact']!=artifact or bundle['foundation_sha256']!=run['foundation_sha256']:
            raise ValueError('Wrong target adapter bundle')
        self.predictor.model.sam_prompt_encoder.load_state_dict(bundle['prompt_encoder'],strict=True)
        self.predictor.model.sam_mask_decoder.load_state_dict(bundle['mask_decoder'],strict=True)
        self.predictor.model.requires_grad_(False)
        self.predictor.model.eval()
        self.source=None
        self.provenance=dict(backend='public_manual_OCTA_'+target+'_specialized_native_SAM2',
            modality='OCTA',segmentation_target=target,model_is_Meta_SAM=True,
            foundation_sha256=run['foundation_sha256'],adapter_sha256=receipt['checkpoint_sha256'],
            training_manifest_sha256=run['contract']['manifest_sha256'],
            trained_on_public_manual_masks=True,pseudo_masks_used=False,
            native_point_box_mask_prompt_supported=True,multimask_output=False,
            image_embedding_input='verified original source RGB, never intermediate actor resize',
            actor_grid_hw=[SIZE,SIZE],actor_mask_projection='nearest-neighbor from native predicted mask',
            candidate_selection='single supervised output; no gold-based selection',
            FAZ_supported=target=='faz',artery_vein_supported=False,clinical_release=False)

    def bind_record(self,source):
        if (set(source)!={'image','image_sha256','modality','segmentation_target'}
                or source['modality']!='OCTA' or source['segmentation_target']!=self.target):
            raise ValueError('OCTA SAM2 tool requires an OCTA retinal-vessel source binding')
        if digest(source['image'])!=source['image_sha256']:
            raise ValueError('OCTA source changed')
        self.source=dict(source)

    def set_image(self,image):
        if self.source is None or image.size!=(SIZE,SIZE):
            raise ValueError('Bind an original OCTA source and declare the actor geometry')
        from PIL import Image
        if digest(self.source['image'])!=self.source['image_sha256']:
            raise ValueError('Original OCTA bytes changed after source binding')
        with Image.open(self.source['image']) as original:
            native=original.convert('RGB')
            expected=native.resize((SIZE,SIZE),Image.Resampling.BICUBIC)
        if not np.array_equal(np.asarray(image),np.asarray(expected)):
            raise ValueError('Provided actor image does not match the bound OCTA source')
        self.native_size_wh=native.size
        # Preserve the exact tool preprocessing used in domain pretraining.
        # The MLLM's display grid is not a new acquisition/native source grid.
        super().set_image(native)

    def _predict(self,box,points=None,labels=None,logits=None):
        # The single mask token was explicitly supervised in tool pretraining.
        # Untrained alternate heads must not be selected by a stale IoU head.
        from PIL import Image
        w,h=self.native_size_wh
        scale=np.asarray([w/SIZE,h/SIZE],np.float32)
        with self.torch.inference_mode(),self.torch.autocast('cuda',dtype=self.torch.bfloat16):
            masks,quality,low=self.predictor.predict(box=np.asarray(box,np.float32)*np.tile(scale,2),
                point_coords=None if not points else np.asarray(points,np.float32)*scale,
                point_labels=None if not labels else np.asarray(labels,np.int32),
                mask_input=logits,multimask_output=False)
        if masks.shape!=(1,h,w) or low.shape!=(1,256,256) or not np.isfinite(low).all() or not np.isfinite(quality).all():
            raise ValueError('Invalid actual native OCTA SAM2 output')
        self.last_native_mask=masks[0].astype(bool)
        display=np.asarray(Image.fromarray(self.last_native_mask.astype(np.uint8)*255).resize(
            (SIZE,SIZE),Image.Resampling.NEAREST))>0
        return display,low.copy()
