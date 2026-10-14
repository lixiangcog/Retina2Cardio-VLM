"""Report-guided visual evidence acquisition, independent of clinical labels.

An executable interaction contract, NOT a claim that a policy has been trained.
The policy proposes questions/actions; dedicated tools return real masks; only
deterministic measurements enter report claims. No diagnostic reward is used.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from retina2cardio.inputs import validate, source_key

ROLES = ('CFP', 'DCP', 'SCP', 'RPCP', 'OCTA_UNKNOWN')

VERSION = 'report_guided_visual_evidence_v53'
TARGETS = {
    'retinal_vessels': {'roles': ROLES, 'metrics': ['predicted_vessel_area_fraction',
        'predicted_centerline_fraction','predicted_width_proxy_mean_grid_px']},
    'faz': {'roles': ('DCP', 'SCP'), 'metrics': ['predicted_faz_area_fraction','predicted_faz_perimeter_over_sqrt_area']},
    'optic_disc': {'roles': ('CFP',), 'metrics': ['predicted_disc_area_fraction']},
}
LIMITATION = ('Algorithmic mask geometry only; historical layer mapping is not independently '
              'device-verified. No physical calibration or normal reference interval is available. '
              'These observations cannot independently diagnose CAD.')


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def decode(text):
    # No extracting a convenient JSON substring or silently repairing actions.
    value = json.loads(text)
    if not isinstance(value, dict): raise ValueError('One JSON object required')
    return value


def request_key(action):
    return json.dumps({k: action[k] for k in ('role', 'target', 'metric', 'roi')}, sort_keys=True)


def available_targets(row):
    """Intersect anatomical tool permissions with the supplied original views."""
    roles = {v['role'] for v in row['views']}
    return {name: dict(target, roles=[r for r in target['roles'] if r in roles])
            for name, target in TARGETS.items() if roles.intersection(target['roles'])}


def validate_action(action, row, evidence, max_calls=3):
    if action.get('action') == 'finish':
        if set(action) != {'action', 'reason'} or not isinstance(action['reason'], str) or not action['reason'].strip():
            raise ValueError('Explicit stop reason required')
        return
    required = {'action', 'observation_id', 'question', 'role', 'source_sha256', 'target', 'metric', 'roi'}
    if set(action) != required or action['action'] != 'acquire': raise ValueError('Invalid acquisition schema')
    if len(evidence) >= max_calls: raise ValueError('Acquisition budget exhausted')
    if not all(isinstance(action[k], str) and action[k].strip() for k in ('observation_id', 'question')):
        raise ValueError('A report observation and a question are required')
    target = available_targets(row).get(action['target'])
    if not target or action['role'] not in target['roles'] or action['metric'] not in target['metrics']:
        raise ValueError('Wrong target/layer/metric; RPCP is not a macular FAZ input')
    view = next((v for v in row['views'] if v['role'] == action['role']), None)
    if view is None: raise ValueError('Requested layer was not supplied')
    if action['source_sha256'] != view['sha256']: raise ValueError('Hidden/other source image requested')
    roi = action['roi']
    if not isinstance(roi, list) or len(roi) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in roi):
        raise ValueError('ROI must be four finite normalized coordinates')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1): raise ValueError('Invalid normalized ROI')
    if any(e['request_key'] == request_key(action) for e in evidence):
        raise ValueError('Repeated acquisition does not create new evidence')


def policy_prompt(stage, row, state):
    validate(row)
    order = ', '.join(v['role'] for v in row['views'])
    unknown = any(v['role'] == 'OCTA_UNKNOWN' for v in row['views'])
    layer_note = (' OCTA_UNKNOWN is one supplied flow-en-face image with unverified layer identity. '
                  'Do not infer DCP/SCP/RPCP, request macular FAZ, or imply three OCTA layers are available.'
                  if unknown else '')
    common = ('<image>\nOrdered supplied images: ' + order + '. ' + layer_note + ' Clinical inputs: ' +
              json.dumps(row['clinical'], ensure_ascii=False) + '\n' + LIMITATION +
              '\nSources: ' + json.dumps([{k: v[k] for k in ('role', 'sha256')} for v in row['views']]) +
              '\nCurrent observations and ACTUAL tool returns: ' + json.dumps(state, ensure_ascii=False))
    if stage == 'plan':
        return common + ('\nReturn ONLY JSON. Choose one clinically relevant observation to verify, '
            'not an assumed diagnosis. Schema: {"action":"acquire","observation_id":"O1",'
            '"question":"what visible observation needs measurement?","role":"SCP",'
            '"source_sha256":"exact supplied hash","target":"retinal_vessels",'
            '"metric":"predicted_vessel_area_fraction","roi":[0,0,1,1]}. '
            'ROI is a rectangular measurement region in the original image normalized to [0,1], '
            'not a verified valid imaging field. Allowed target/metric/role registry: ' +
            json.dumps(available_targets(row)) + '. The schema example is illustrative; select a role '
            'actually supplied in the registry. Never change the ROI to make results abnormal. '
            'If evidence is sufficient, unavailable, or the budget is reached, return '
            '{"action":"finish","reason":"specific reason"}. At most three distinct requests.')
    if stage != 'report': raise ValueError('Unknown policy stage')
    return common + ('\nReturn ONLY JSON: {"screening_probability":0.5,"claims":['
        '{"observation_id":"O1","disposition":"revise","evidence_id":"E01",'
        '"metric":"exact returned metric","value":0.12,"unit":"fraction",'
        '"interpretation":"descriptive_only"}],"report":"four sections: Imaging findings; '
        'Evidence; CAD screening; Limitations"}. For unmeasurable evidence use disposition '
        '"uncertain", value null, metric and unit null. Allowed dispositions: retain/revise/remove/uncertain. '
        'Every quantitative claim must quote an accepted evidence value exactly. '
        'Do not claim reduced/increased/abnormal or physical units without reference/calibration. '
        'Classification is a research screening prediction, not a diagnosis. '
        'Keep qualitative observations separate from quantitative claims; cite [E01] in report text. '
        'If no evidence, claims may be empty. Explain uncertainty and any correction to initial observations.')


def measure(action, mask, status, trace, provenance, out):
    """Persist actual masks, ROI, revisions and a reproducible pixel statistic."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    result = dict(request_key=request_key(action), request=copy.deepcopy(action),
        source_sha256=action['source_sha256'], role=action['role'], target=action['target'],
        status=status, metric=action['metric'], value=None, unit='fraction',
        technical_gate=None, clinical_measurement=False, physical_calibration_verified=False,
        independently_validated_mask=False, tool=provenance, limitation=LIMITATION,
        measurement_scope='whole_image_grid' if action['roi'] == [0, 0, 1, 1] else 'rectangular_ROI',
        corrected=False, mask_sha256=None, mask_steps=[], initial_mask_sha256=None)
    if mask is None:
        result['technical_gate'] = status
    else:
        if mask.dtype != bool or mask.ndim != 2: raise ValueError('Boolean mask required')
        if status=='accepted' and not trace:raise ValueError('Accepted agent mask requires actual returned-mask history')
        if trace and not np.array_equal(trace[-1],mask):raise ValueError('Final mask differs from recorded last tool return')
        for i, item in enumerate(trace):
            if item.shape != mask.shape or item.dtype != bool: raise ValueError('Trace shape differs')
            p = out / f'mask_step_{i:02d}.png'; Image.fromarray(item.astype(np.uint8)*255).save(p)
            result['mask_steps'].append(dict(path=str(p.resolve()), sha256=sha_file(p)))
        final = out / 'final_mask.png'; Image.fromarray(mask.astype(np.uint8)*255).save(final)
        result.update(mask_path=str(final.resolve()), mask_sha256=sha_file(final), grid_hw=list(mask.shape))
        if trace:
            result['initial_mask_sha256'] = result['mask_steps'][0]['sha256']
            result['corrected'] = any(not np.array_equal(trace[0], t) for t in trace[1:])
        h, w = mask.shape; x0, y0, x1, y1 = action['roi']
        bounds = [int(math.floor(x0*w)), int(math.floor(y0*h)), int(math.ceil(x1*w)), int(math.ceil(y1*h))]
        roi = np.zeros_like(mask); roi[bounds[1]:bounds[3], bounds[0]:bounds[2]] = True
        rp = out / 'measurement_roi.png'; Image.fromarray(roi.astype(np.uint8)*255).save(rp)
        result.update(roi_path=str(rp.resolve()), roi_sha256=sha_file(rp), roi_pixel_bounds=bounds,
                      roi_pixel_count=int(roi.sum()), calculator_version=VERSION)
        gate = None
        if not mask.any(): gate = 'empty_mask'
        elif action['target'] == 'retinal_vessels' and mask.mean() > .45: gate = 'dense_region_not_vessels'
        elif action['target'] != 'retinal_vessels':
            from scipy.ndimage import label
            if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any(): gate = 'region_border_contact'
            elif label(mask, np.ones((3,3)))[1] != 1: gate = 'disconnected_region'
            elif (mask & ~roi).any(): gate = 'region_clipped_by_measurement_ROI'
        result['technical_gate'] = gate or (None if status == 'accepted' else status)
        if result['technical_gate'] is None:
            result.update(value=float((mask & roi).sum()/roi.sum()), foreground_pixels=int((mask & roi).sum()))
            if action['metric'] in ('predicted_centerline_fraction','predicted_width_proxy_mean_grid_px'):
                from retina2cardio.segmentation.metrics import skeletonize
                from scipy.ndimage import distance_transform_edt
                center=skeletonize(mask) & roi
                if not center.any():
                    result.update(value=None,technical_gate='no_centerline_in_ROI')
                elif action['metric']=='predicted_centerline_fraction':result['value']=float(center.sum()/roi.sum())
                else:
                    result.update(value=float((2*distance_transform_edt(mask))[center].mean()),unit='resampled_grid_px')
            elif action['metric']=='predicted_faz_perimeter_over_sqrt_area':
                padded=np.pad(mask.astype(np.int8),1)
                perimeter=np.abs(np.diff(padded,axis=0)).sum()+np.abs(np.diff(padded,axis=1)).sum()
                result.update(value=float(perimeter/np.sqrt(mask.sum())),unit='dimensionless_pixel_grid')
    dump(out/'measurement.json', result)
    result['tool_receipt_sha256'] = sha_file(out/'measurement.json')
    result['measurement_receipt'] = str((out/'measurement.json').resolve())
    return result


def validate_report(report, evidence):
    if set(report) != {'screening_probability', 'claims', 'report'}: raise ValueError('Invalid report schema')
    p = report['screening_probability']
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1: raise ValueError('Invalid probability')
    if not isinstance(report['report'], str) or not report['report'].strip(): raise ValueError('Empty report')
    if not isinstance(report['claims'], list): raise ValueError('Claims must be a list')
    by_id = {e['evidence_id']: e for e in evidence}; cited = set()
    for c in report['claims']:
        if set(c) != {'observation_id','disposition','evidence_id','metric','value','unit','interpretation'}:
            raise ValueError('Invalid claim schema')
        e = by_id.get(c['evidence_id'])
        if e is None or c['observation_id'] != e['request']['observation_id']: raise ValueError('Claim/source mismatch')
        if c['evidence_id'] in cited: raise ValueError('Repeated claim')
        cited.add(c['evidence_id'])
        if c['disposition'] not in ('retain','revise','remove','uncertain'): raise ValueError('Invalid revision')
        if c['interpretation'] != 'descriptive_only': raise ValueError('Unsupported clinical normality inference')
        if c['value'] is None:
            if c['metric'] is not None or c['unit'] is not None or c['disposition'] not in ('remove','uncertain'):
                raise ValueError('Uncertainty must not imply a measurement')
        elif (e['status'] != 'accepted' or e['technical_gate'] is not None or type(c['value']) not in (int,float)
              or not math.isfinite(c['value']) or c['value'] != e['value']
              or c['metric'] != e['metric'] or c['unit'] != e['unit']):
            raise ValueError('Unaccepted, invented or misquoted measurement')
        if '['+c['evidence_id']+']' not in report['report']: raise ValueError('Missing visible evidence citation')
    if cited != set(by_id): raise ValueError('Tool results must be addressed, including failures')
    # This checks structured facts, not every semantic statement in free text.
    return dict(structured_claims_verified=True, free_text_clinical_review_complete=False)


def trajectory_reward(report, evidence, invalid_actions=0):
    """Deterministic grounding reward, NOT clinical quality or CAD correctness.

    No CAD label argument, no reward for abnormality, no penalty for warranted
    uncertainty. Semantic relevance needs independently reviewed supervision.
    """
    try: validate_report(report, evidence)
    except (ValueError, KeyError, TypeError): return -1.0
    return 1.0 - .03 * len(evidence) - .25 * invalid_actions


def learning_reward(episode, reviewed_tasks):
    """Training-only task coverage + factual grounding, with explicit limits.

    Reviewed tasks are teacher supervision, never given to the rollout policy.
    This does not assess free-text clinical truth or prove a CAD benefit.
    All-stop cannot beat addressing a reviewed evidence question. A technically
    unmeasurable requested task counts as addressed, so there is no incentive
    to force mask acceptance or fabricate a metric.
    """
    if not reviewed_tasks: raise ValueError('Reviewed task supervision required before report GRPO')
    if any(t.get('truncated',False) for t in episode['turns']): return -1.
    needed={(t['role'],t['target'],t['metric'],tuple(t['roi'])) for t in reviewed_tasks}
    observed={(e['role'],e['target'],e['metric'],tuple(e['request']['roi'])) for e in episode['evidence']}
    coverage=len(needed & observed)/len(needed)
    try: validate_report(episode['report'],episode['evidence'])
    except (ValueError,KeyError,TypeError): return -1.
    return coverage - .03*len(observed) - .15*len(observed-needed)


def run_episode(row, policy, backend, output, max_calls=3):
    validate(row); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    state = dict(evidence=[], max_calls=max_calls); turns=[]; stop='budget'; invalid=0
    def record(stage, raw):
        turn=dict(stage=stage,state=copy.deepcopy(state),raw=raw)
        if hasattr(policy,'last_response_ids'):
            turn.update(response_ids=list(policy.last_response_ids),truncated=policy.last_response_truncated)
        turns.append(turn)
    for _ in range(max_calls+1):
        raw = policy('plan', row, copy.deepcopy(state)); record('plan',raw)
        try:
            action = decode(raw); validate_action(action, row, state['evidence'], max_calls)
        except (ValueError, KeyError, TypeError) as exc:
            invalid += 1; stop='invalid_action'; turns[-1]['error']=str(exc); break
        if action['action'] == 'finish': stop=action['reason']; break
        index=len(state['evidence'])+1
        response=backend(row, action, output/f'E{index:02d}')
        response['evidence_id']=f'E{index:02d}'; state['evidence'].append(response)
    raw=policy('report',row,copy.deepcopy(state));record('report',raw)
    report=None; qc=None; error=None
    try:
        report=decode(raw);qc=validate_report(report,state['evidence'])
    except (ValueError,KeyError,TypeError) as exc: error=str(exc)
    result=dict(schema=VERSION,id=row['id'],input_source_key=source_key(row),turns=turns,
        evidence=state['evidence'],report=report,report_qc=qc,report_error=error,stop_reason=stop,
        reward=trajectory_reward(report,state['evidence'],invalid) if report is not None else -1.,
        clinical_quality_verified=False,labels_loaded=False,actual_optimizer_update=False)
    dump(output/'episode.json',result);return result
