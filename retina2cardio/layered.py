"""Ordered four-view input validation, no sample construction."""
import hashlib,json,math
ROLES=("CFP","DCP","SCP","RPCP")

def source_key(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

def validate(row):
    if set(row) != {'id', 'views', 'clinical', 'biomarker_evidence', 'prompt'}:
        raise ValueError('Unexpected input fields, including supervision or copied reports')
    views = row['views']
    if len(views) != 4 or tuple(v['role'] for v in views) != ROLES:
        raise ValueError('Exactly CFP, DCP, SCP, RPCP in that order are required')
    if len({v['sha256'] for v in views}) != 4:
        raise ValueError('Four distinct source images required')
    for v in views:
        if set(v) != {'path', 'modality', 'sha256', 'role'}:
            raise ValueError('Unexpected view schema')
        if v['modality'] != ('CFP' if v['role'] == 'CFP' else 'OCTA'):
            raise ValueError('Image role/modality mismatch')
        if not v['path'] or v['path'].startswith(('/', '\\')) or '..' in v['path'].split('/') or ':' in v['path']:
            raise ValueError('Relative source path required')
        if len(v['sha256']) != 64:
            raise ValueError('Missing source SHA256')
    hashes = {v['role']: v['sha256'] for v in views}
    for item in row['biomarker_evidence']:
        if item['role'] not in hashes or item['source_sha256'] != hashes[item['role']]:
            raise ValueError('Biomarker belongs to another image')
        if item.get('status') != 'accepted' or not item.get('metric') or not item.get('unit'):
            raise ValueError('Only accepted, named, unit-bearing measurements are input evidence')
        value = item.get('value')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid biomarker value')
        if not item.get('mask_sha256') or not item.get('tool_receipt_sha256'):
            raise ValueError('Mask and tool receipt provenance required')
        if item['unit'] in {'mm', 'mm2', 'um', 'clinical_percent'} and not item.get('physical_calibration_verified'):
            raise ValueError('Uncalibrated geometry cannot use physical/clinical units')
    if row['prompt'].count('<image>') != 1:
        raise ValueError('One placeholder for the ordered four-image sequence required')
