"""Method-level optimizer steps; callers supply their own reviewed trajectories.

No data construction, dataset readers, folds, test selection, or experiment loops.
"""
import json
from .evidence import validate_report,learning_reward,validate_action,sha_file,source_key
from .segmentation.optimization import advantages,clipped_loss


def _update(parameters,optimizer,losses):
    import torch
    parameters=list(parameters);before=[p.detach().cpu().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    for loss in losses:
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
    optimizer.step()
    changed=sum(not torch.equal(p.detach().cpu(),b) for p,b in zip(parameters,before))
    if not changed:raise ValueError('No actual parameter change; do not claim a trained update')
    return dict(changed_parameter_tensors=changed,gradient_norm=float(norm))


def report_sft_step(policy,optimizer,row,episode,weight=1.):
    """Supervise actual action/return/report turns; segmentation stays frozen."""
    if weight<=0:raise ValueError('Positive weighting required')
    if episode['input_source_key']!=source_key(row) or episode.get('synthetic_fixture',False):raise ValueError('Real matching teacher episode required')
    validate_report(episode['report'],episode['evidence'])
    turns=episode['turns']
    if not turns or turns[-1]['stage']!='report':raise ValueError('Complete teacher trajectory required')
    seen=[]
    for turn in turns:
        if turn['state'].get('evidence')!=seen:raise ValueError('Future/unexecuted teacher evidence')
        response=json.loads(turn['raw'])
        if turn['stage']=='plan':
            validate_action(response,row,seen)
            if response['action']=='acquire':
                evidence=episode['evidence'][len(seen)]
                if evidence['request']!=response:raise ValueError('Teacher action differs from real request')
                if sha_file(evidence['measurement_receipt'])!=evidence['tool_receipt_sha256']:raise ValueError('Actual measurement receipt changed')
                from pathlib import Path
                actual=json.loads(Path(evidence['measurement_receipt']).read_text())
                for key in ('source_sha256','role','target','metric','value','mask_sha256','roi_sha256'):
                    if actual.get(key)!=evidence.get(key):raise ValueError('Teacher measurement not grounded')
                for key in ('mask','roi'):
                    if evidence.get(key+'_path') and sha_file(evidence[key+'_path'])!=evidence[key+'_sha256']:raise ValueError('Actual mask/ROI changed')
                seen.append(evidence)
        elif turn['stage']!='report' or turn is not turns[-1] or response!=episode['report']:raise ValueError('Invalid final supervised report')
    if seen!=episode['evidence']:raise ValueError('Unexecuted evidence')
    losses=(-policy.logp(t['stage'],row,t['state'],t['raw']).mean()*weight/len(turns) for t in turns)
    return _update((p for p in policy.model.parameters() if p.requires_grad),optimizer,losses)


def report_grpo_step(policy,optimizer,row,episodes,reviewed_tasks):
    """Eight genuine tool-report rollouts; no CAD-test or invented clinical reward."""
    import torch
    if len(episodes)!=8:raise ValueError('Exactly eight real rollouts required')
    rewards=[learning_reward(ep,reviewed_tasks) for ep in episodes];adv=advantages(rewards,[0]*8);cached=[]
    with torch.no_grad():
        for ep in episodes:
            pairs=[]
            for t in ep['turns']:
                old=policy.logp(t['stage'],row,t['state'],t['raw'],t['response_ids']).detach().cpu()
                policy.model.set_adapter('reference')
                try:ref=policy.logp(t['stage'],row,t['state'],t['raw'],t['response_ids']).detach().cpu()
                finally:policy.model.set_adapter('default')
                pairs.append((old,ref))
            cached.append(pairs)
    def losses():
        for k,ep in enumerate(episodes):
            for t,(old,ref) in zip(ep['turns'],cached[k]):
                logp=policy.logp(t['stage'],row,t['state'],t['raw'],t['response_ids'])
                loss,_=clipped_loss(logp,old.to(logp.device),ref.to(logp.device),adv[k])
                yield loss/(8*len(ep['turns']))
    result=_update((p for p in policy.model.parameters() if p.requires_grad),optimizer,losses())
    return dict(result,rewards=rewards,clinical_accuracy_reward_used=False,segmentation_policy_updated=False)


def segmentation_grpo_step(actor,optimizer,prompt_groups):
    """Update only low-level actor LoRA using independently annotated mask tasks.

    Each supplied group contains inputs, eight sampled response objects and their
    actual target-specific rewards. Constructing those annotations is external.
    """
    rollouts=[];rewards=[];groups=[]
    for group,batch in enumerate(prompt_groups):
        if len(batch['responses'])!=8 or len(batch['rewards'])!=8:raise ValueError('Eight response/reward pairs required')
        for item,reward in zip(batch['responses'],batch['rewards']):
            old,ref=actor.old_and_ref(batch['inputs'],item['ids'])
            rollouts.append((batch['inputs'],item['ids'],old,ref));rewards.append(0. if item['truncated'] else reward);groups.append(group)
    adv=advantages(rewards,groups)
    def losses():
        for k,(inputs,ids,old,ref) in enumerate(rollouts):
            current=actor.logp(inputs,ids)
            loss,_=clipped_loss(current,old.to(current.device),ref.to(current.device),adv[k]);yield loss/len(rollouts)
    return dict(_update(actor.parameters,optimizer,losses()),CAD_model_updated=False,specialist_updated=False)
