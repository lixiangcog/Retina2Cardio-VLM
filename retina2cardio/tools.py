"""Actual specialist/GRPO backend for report-guided requests. No labels loaded."""
import json
from pathlib import Path
from retina2cardio.evidence import measure, dump, sha_file, validate_action


class SpecialistBackend:
    def __init__(self, image_root, registry, output_root):
        self.root=Path(image_root).resolve();self.registry=registry;self.output_root=Path(output_root)
        self.instances={}

    def __call__(self, row, action, output):
        validate_action(action,row,[])
        view=next(v for v in row['views'] if v['role']==action['role'])
        src=(self.root/view['path']).resolve()
        if not src.is_relative_to(self.root) or sha_file(src)!=view['sha256']:raise ValueError('Original image binding changed')
        key=view['modality']+':'+action['target'];entry=self.registry.get('tools',{}).get(key)
        if entry is None:return measure(action,None,'target_specialist_not_configured',[],{},output)
        # A QC receipt must explicitly cover this source and this target. A vessel
        # source review is not silently promoted to FAZ/disc anatomical approval.
        qc=entry.get('source_qc',{}).get(view['sha256'])
        if not qc or qc.get('eligible') is not True:
            return measure(action,None,'source_target_QC_missing_or_rejected',[],{'profile':entry['profile']},output)
        qp=Path(qc['receipt'])
        if sha_file(qp)!=qc['receipt_sha256']:raise ValueError('Technical source QC changed')
        source_qc=json.loads(qp.read_text())
        if not any(x.get('source_sha256')==view['sha256'] and x.get('target')==action['target'] and x.get('eligible') is True
                   for x in source_qc.get('reviews',[])):raise ValueError('Receipt does not attest this source/target')
        if key not in self.instances:
            from types import SimpleNamespace
            from retina2cardio.segmentation.runtime import create_tool,NativeActor,PROMPT_CONTRACT
            profile=entry['profile'];ck=Path(entry['policy_checkpoint'])
            update=json.loads((ck/'update_receipt.json').read_text())
            run=json.loads((ck.parent/'run.json').read_text())
            if update.get('changed_parameter_tensors',0)<=0 or run['args']['tool_profile']!=profile:
                raise ValueError('Trained target-specific GRPO policy required')
            if run['prompt_contract']!=PROMPT_CONTRACT:raise ValueError('Policy/prompt mismatch requires reevaluation')
            # Require an explicit public-only promotion receipt, never CAD score-based selection.
            promotion=json.loads(Path(entry['promotion_receipt']).read_text())
            if (promotion.get('passed') is not True or promotion.get('cad_test_used') is not False
                or promotion.get('adapter_sha256')!=sha_file(ck/'adapter_model.safetensors')
                or promotion.get('target')!=action['target']):raise ValueError('Public target validation gate not met')
            evidence_path=Path(promotion['public_validation_receipt'])
            if sha_file(evidence_path)!=promotion['public_validation_sha256']:raise ValueError('Public validation receipt changed')
            tool=create_tool(SimpleNamespace(tool_profile=profile,cfp_runtime=entry.get('cfp_runtime'),
                octa_runtime=entry.get('octa_runtime'),sam_checkpoint=entry.get('sam_checkpoint')),
                self.output_root/(key.replace(':','_')+'_tool_cache'))
            if tool.provenance!=run['mask_tool']:raise ValueError('Frozen specialist differs from policy training')
            actor=NativeActor(entry['actor'],adapter=str(ck))
            self.instances[key]=(actor,tool,sha_file(ck/'adapter_model.safetensors'))
        actor,tool,adapter_sha=self.instances[key]
        from retina2cardio.segmentation.episode import run_source_episode,run_octa_source_episode,run_region_source_episode
        query={'retinal_vessels':'all visible retinal blood vessels','faz':'the foveal avascular zone (FAZ)',
               'optic_disc':'the optic disc'}[action['target']]
        source=dict(image=str(src),image_sha256=view['sha256'],modality=view['modality'],
                    query=query,segmentation_target=action['target'])
        # ROI restricts measurement, not a diagnosis-driven mask alteration.
        # Keep the original validated low-level prompt; the high-level question is
        # linked in the request and is NOT used to encourage abnormal segmentation.
        fn=(run_region_source_episode if action['target']!='retinal_vessels' else
            run_octa_source_episode if view['modality']=='OCTA' else run_source_episode)
        trace=[];mask,actions,status=fn(actor,tool,source,mask_trace=trace)
        result=measure(action,mask,status,trace,dict(tool.provenance,agent_adapter_sha256=adapter_sha),output)
        dump(Path(output)/'controller_actions.json',actions)
        result['actions_receipt_sha256']=sha_file(Path(output)/'controller_actions.json')
        # Persist an actual mask overlay, not an illustrative replacement.
        if mask is not None:
            from PIL import Image
            from retina2cardio.segmentation.runtime import overlay
            with Image.open(src) as im:preview=im.convert('RGB').resize((mask.shape[1],mask.shape[0]),Image.Resampling.BICUBIC)
            overlay(preview,mask).save(Path(output)/'final_mask_overlay.png')
        return result
