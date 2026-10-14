"""Use the existing author's OD/OC head for the whole optic disc only."""
import numpy as np
from retina2cardio.segmentation.cfp_specialist import CFPAuthorSpecialist, restore_crop
from retina2cardio.utils import digest


class CFPDiscSpecialist(CFPAuthorSpecialist):
    def __init__(self, runtime, device='cuda'):
        super().__init__(runtime, device)
        self.provenance.update(segmentation_target='optic_disc', modality='CFP',
            author_target_head='od_oc', whole_disc_decode='class 1 OR class 2',
            optic_cup_released=False, clinical_domain_validation=False)

    def candidate(self, job):
        if job['modality'] != 'CFP' or digest(job['image_path']) != job['source_sha256']:
            raise ValueError('Verified original CFP required')
        prepared = self.processor.load_and_preprocess(job['image_path'])
        with self.torch.inference_mode():
            outputs = self.model(prepared['tensor'].to(self.device))
            heads = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = heads[1]
            if tuple(logits.shape) != (1, 3, 640, 640) or not self.torch.isfinite(logits).all():
                raise ValueError('Expected author background/disc/cup head')
            filtered = self.noise_filter.filter_prediction(logits, 'od_oc')
            classes = filtered.argmax(dim=1)[0].cpu().numpy()
        # Cup is anatomically INSIDE the disc. Class 1 alone would produce a rim.
        mask = restore_crop(np.isin(classes, [1, 2]), prepared['crop_coords'], prepared['original_shape'])
        return {'mask': mask, 'provenance': {**self.provenance,
            'source_sha256': job['source_sha256'], 'source_grid_wh': list(prepared['original_shape'])}}, {
            'tool': 'RetSAM_optic_disc_head', 'clinical_release': False}
