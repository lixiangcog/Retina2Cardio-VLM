"""Text Qwen3 with the existing frozen RETFound branches and visual routing."""
from types import SimpleNamespace
import torch
from transformers import AutoConfig,AutoTokenizer,Qwen3ForCausalLM,Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
from .vision.meta import LlavaMetaModel,LlavaMetaForCausalLM
from .vision.optional import OptionalModalityQwen
from .model_utils import core_loading_guard

class RetinalQwen3Model(LlavaMetaModel,Qwen3Model):pass

class RetinalQwen3ForCausalLM(Qwen3ForCausalLM,LlavaMetaForCausalLM):
    def __init__(self,config):
        Qwen3PreTrainedModel.__init__(self,config)
        self.model=RetinalQwen3Model(config)
        self.vocab_size=config.vocab_size
        self.lm_head=torch.nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.post_init()
    def get_model(self):return self.model
    prepare_inputs_labels_for_multimodal=OptionalModalityQwen.prepare_inputs_labels_for_multimodal
    def forward(self,input_ids=None,attention_mask=None,position_ids=None,past_key_values=None,
                inputs_embeds=None,labels=None,images=None,octs=None,image_sizes=None,**kwargs):
        if inputs_embeds is None:
            input_ids,position_ids,attention_mask,past_key_values,inputs_embeds,labels=self.prepare_inputs_labels_for_multimodal(
                input_ids,position_ids,attention_mask,past_key_values,labels,octs,images,image_sizes)
        return Qwen3ForCausalLM.forward(self,input_ids=input_ids,attention_mask=attention_mask,
            position_ids=position_ids,past_key_values=past_key_values,inputs_embeds=inputs_embeds,labels=labels,**kwargs)
    @torch.no_grad()
    def generate(self,inputs=None,images=None,octs=None,**kwargs):
        if 'inputs_embeds' in kwargs:raise ValueError('Use original image routing')
        mask=kwargs.pop('attention_mask',None);kwargs.pop('position_ids',None)
        if images is not None or octs is not None:
            _,_,mask,_,emb,_=self.prepare_inputs_labels_for_multimodal(inputs,None,mask,None,None,octs,images)
        else:emb=self.model.embed_tokens(inputs)
        return Qwen3ForCausalLM.generate(self,inputs_embeds=emb,attention_mask=mask,**kwargs)

def build_text_qwen3(model_base,vision_weights,max_length=2048):
    config=AutoConfig.from_pretrained(model_base,local_files_only=True)
    if config.model_type!='qwen3' or getattr(config,'quantization_config',None):
        raise ValueError('Unquantized text Qwen3 required')
    config.tokenizer_padding_side='right';config.tokenizer_model_max_length=max_length;config.use_cache=False
    model,info=RetinalQwen3ForCausalLM.from_pretrained(model_base,config=config,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='balanced',
        max_memory={0:'10GiB',1:'10GiB'},output_loading_info=True)
    core_loading_guard(info)
    if any(str(v) in ('cpu','disk') for v in model.hf_device_map.values()):raise ValueError('Dual-GPU placement required')
    args=SimpleNamespace(vision_tower=vision_weights,pretrain_mm_mlp_adapter=None,
        mm_vision_select_layer=-2,mm_vision_select_feature='patch',mm_projector_type='mlp2x_gelu',
        mm_patch_merge_type='flat',oct_2d_tower=vision_weights,pretrain_oct_2d_projector=None,oct_2d_projector_type='linear')
    model.model.initialize_vision_modules(args);model.model.initialize_oct_2d_modules(args)
    model.config.mm_use_im_start_end=False;model.config.mm_use_im_patch_token=False;model.config.image_aspect_ratio='ori'
    for module in (model.get_vision_tower(),model.get_oct_2d_tower(),model.model.mm_projector,model.model.oct_2d_projector):
        module.to(device='cuda:0',dtype=torch.bfloat16)
    for tower in (model.get_vision_tower(),model.get_oct_2d_tower()):tower.requires_grad_(False);tower.eval()
    tokenizer=AutoTokenizer.from_pretrained(model_base,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token='<|endoftext|>'
    tokenizer.padding_side='right'
    return model,tokenizer
