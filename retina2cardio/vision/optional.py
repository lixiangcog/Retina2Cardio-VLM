import torch
from transformers import Qwen2ForCausalLM
from .routing import branch_features,pack,IGNORE_INDEX,IMAGE_TOKEN_INDEX

class OptionalModalityQwen:
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask,
                                             past_key_values, labels, octs, images, image_sizes=None):
        if input_ids is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if not (input_ids == IMAGE_TOKEN_INDEX).any():
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            raise ValueError('Unexpected image placeholder inside cached continuation')
        batch = input_ids.shape[0]
        cfp = branch_features(images, batch, self.encode_images)
        octa = branch_features(octs, batch, self.encode_oct_2d_images)
        features = []
        for a, b in zip(cfp, octa):
            present = [v for v in (a, b) if v is not None]
            features.append(torch.cat(present, dim=0) if present else None)
        emb, mask, pos, gold = pack(input_ids, attention_mask, labels, features,
                                   self.get_model().embed_tokens,
                                   self.config.tokenizer_model_max_length,
                                   getattr(self.config, 'tokenizer_padding_side', 'right'))
        return None, pos, mask, past_key_values, emb, gold

    @torch.no_grad()
    def generate(self, inputs=None, images=None, octs=None, **kwargs):
        if 'inputs_embeds' in kwargs:
            raise ValueError('Use the same image-only route as training')
        attention_mask = kwargs.pop('attention_mask', None)
        kwargs.pop('position_ids', None)
        if images is not None or octs is not None:
            _, _, mask, _, emb, _ = self.prepare_inputs_labels_for_multimodal(
                inputs, None, attention_mask, None, None, octs, images)
        else:
            emb = self.get_model().embed_tokens(inputs)
            mask = attention_mask
        # Image features have already been injected. Cached decoding is text-only.
        return Qwen2ForCausalLM.generate(self, inputs_embeds=emb, attention_mask=mask, **kwargs)
