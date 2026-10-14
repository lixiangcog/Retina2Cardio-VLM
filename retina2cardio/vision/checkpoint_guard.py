"""Fail closed on incomplete weights used by the legacy patch-token forward.

No normalization or feature definition is changed. The legacy forward does not
use head/fc_norm, so their absent weights are explicitly allowed. This guard is
NOT suitable for the normalized Stage-II forward, which has its own strict loader.
"""
from collections.abc import Mapping


def load_patch_encoder_weights(model, checkpoint):
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping):
        raise ValueError('RETFound checkpoint must contain a state dictionary')
    expected = model.state_dict()
    required = {k for k in expected if k in {'cls_token', 'pos_embed'}
                or k.startswith(('patch_embed.', 'blocks.'))}
    if not required or not any(k.startswith('blocks.') for k in required):
        raise ValueError('Expected the RETFound patch-token encoder architecture')
    missing = sorted(required - set(state))
    mismatched = sorted(k for k in required & set(state)
                        if not hasattr(state[k], 'shape') or tuple(state[k].shape) != tuple(expected[k].shape))
    if missing or mismatched:
        raise ValueError(f'Incomplete RETFound encoder weights: missing={missing}; shape_mismatch={mismatched}')
    report = model.load_state_dict(state, strict=False)
    # Do not rely only on a pre-load key comparison: framework hooks can alter it.
    if required & set(report.missing_keys):
        raise ValueError('Framework did not load all required RETFound weights')
    model.checkpoint_load_audit = {
        'critical_encoder_keys_loaded': len(required),
        'missing_unused_keys': list(report.missing_keys),
        'unexpected_checkpoint_keys': list(report.unexpected_keys),
        'features': 'legacy_unnormalized_patch_tokens_unchanged',
        'normalization_parameters_required': False,
    }
    return report
