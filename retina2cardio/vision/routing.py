"""Pack only observed visual branches; preserve case order and causal masks."""
import torch

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


def branch_features(inputs, batch_size, encode):
    if inputs is None:
        return [None] * batch_size
    if len(inputs) != batch_size:
        raise ValueError('Visual batch length differs from text batch')
    counts, tensors = [], []
    for value in inputs:
        if value is None:
            counts.append(0)
        else:
            value = value.unsqueeze(0) if value.ndim == 3 else value
            if value.ndim != 4 or value.shape[0] < 1:
                raise ValueError('Visual branch must have actual images or None')
            counts.append(value.shape[0])
            tensors.append(value)
    if not tensors:
        return [None] * batch_size
    encoded = encode(torch.cat(tensors, dim=0))
    if encoded.ndim != 3 or encoded.shape[0] != sum(counts):
        raise ValueError('Encoder changed image count or token rank')
    result, offset = [], 0
    for count in counts:
        result.append(encoded[offset:offset+count].flatten(0, 1) if count else None)
        offset += count
    return result


def pack(input_ids, attention_mask, labels, features, embed, max_length, padding_side='right'):
    if len(features) != input_ids.shape[0]:
        raise ValueError('Feature/text batch mismatch')
    mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    target = torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels
    embeddings, targets = [], []
    for ids, valid, gold, visual in zip(input_ids, mask, target, features):
        ids, gold = ids[valid], gold[valid]
        indices = torch.where(ids == IMAGE_TOKEN_INDEX)[0]
        if indices.numel() != 1 or visual is None or not visual.numel():
            raise ValueError('Each multimodal case requires one placeholder and nonempty observed features')
        at = int(indices.item())
        text = embed(torch.cat((ids[:at], ids[at+1:])))
        visual = visual.to(device=text.device, dtype=text.dtype)
        merged = torch.cat((text[:at], visual, text[at:]), dim=0)
        if len(merged) > max_length:
            raise ValueError('Refusing silent visual/target truncation')
        gold = torch.cat((gold[:at], gold.new_full((len(visual),), IGNORE_INDEX), gold[at+1:]))
        embeddings.append(merged)
        targets.append(gold)
    size = max(map(len, embeddings))
    output, masks, positions, outlabels = [], [], [], []
    for emb, gold in zip(embeddings, targets):
        pad = size-len(emb)
        zero = emb.new_zeros((pad, emb.shape[-1]))
        if padding_side == 'left':
            output.append(torch.cat((zero, emb)))
            masks.append(torch.cat((gold.new_zeros(pad), gold.new_ones(len(emb)))).bool())
            positions.append(torch.cat((gold.new_zeros(pad), torch.arange(len(emb), device=gold.device))))
            outlabels.append(torch.cat((gold.new_full((pad,), IGNORE_INDEX), gold)))
        else:
            output.append(torch.cat((emb, zero)))
            masks.append(torch.cat((gold.new_ones(len(emb)), gold.new_zeros(pad))).bool())
            positions.append(torch.cat((torch.arange(len(emb), device=gold.device), gold.new_zeros(pad))))
            outlabels.append(torch.cat((gold, gold.new_full((pad,), IGNORE_INDEX))))
    return (torch.stack(output), torch.stack(masks), torch.stack(positions),
            torch.stack(outlabels) if labels is not None else None)
