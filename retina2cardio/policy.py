"""Actual BF16 text Qwen3 + ordered RETFound policy, no diagnosis prefill."""
import json
from pathlib import Path
from retina2cardio.inputs import VERSION
from retina2cardio.evidence import policy_prompt, sha_file


class ReportPolicy:
    def __init__(self, checkpoint, data, sampling=False, trainable=False):
        import torch
        from peft import PeftModel
        from retina2cardio.qwen3_retinal import build_text_qwen3
        self.torch=torch;self.root=Path(data).resolve();self.sampling=sampling
        ck=Path(checkpoint);config=json.loads((ck/'cad_run_config.json').read_text())
        if config.get('quantized') is not False:
            raise ValueError('BF16 unquantized report checkpoint required')
        model,self.tokenizer=build_text_qwen3(config['model_base'],config['vision_weights'],8192)
        extra=torch.load(ck/'non_lora_trainables.bin',map_location='cpu',weights_only=True)
        expected={n for n,p in model.named_parameters() if '.mm_projector.' in n or '.oct_2d_projector.' in n}
        if set(extra)!=expected:raise ValueError('Retinal projector checkpoint incomplete')
        model.load_state_dict(extra,strict=False)
        self.model=PeftModel.from_pretrained(model,ck,is_trainable=trainable).eval()
        self.base=self.model.get_base_model();self.device=self.base.get_input_embeddings().weight.device
        self.receipt=dict(checkpoint=str(ck),adapter_sha256=sha_file(ck/'adapter_model.safetensors'),
            model='text Qwen3-8B + CFP RETFound + shared OCTA RETFound',quantized=False,
            report_training_completed=config.get('report_training_completed',False),sampling=sampling)
        self.config=config
        if trainable:
            # Report-level GRPO updates only the report/tool-selection policy.
            # Frozen projectors make the reference adapter a genuine fixed policy.
            for name,p in self.model.named_parameters():p.requires_grad_('lora_' in name)
            for m in self.model.modules():
                if isinstance(m,torch.nn.Dropout):m.p=0.
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.model.train();self.base.get_vision_tower().eval();self.base.get_oct_2d_tower().eval()

    def logp(self,stage,row,state,response,response_ids=None):
        from retina2cardio.model_utils import encode_text
        from retina2cardio.model_utils import image_inputs
        rendered=self.tokenizer.apply_chat_template([{'role':'system','content':'You are a retinal imaging research assistant. Follow the JSON tool protocol; do not invent measurements.'},
            {'role':'user','content':policy_prompt(stage,row,state)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=encode_text(self.tokenizer,rendered)
        answer=(response_ids if response_ids is not None else
                self.tokenizer.encode(response+'<|im_end|>',add_special_tokens=False))
        if not answer:raise ValueError('Empty response cannot be optimized')
        ids=self.torch.tensor([prefix+answer],device=self.device)
        targets=self.torch.tensor([[-100]*len(prefix)+answer],device=self.device)
        cfp,octa=image_inputs(row['views'],self.root,self.base)
        visual=lambda x:[x.to('cuda:0',dtype=self.torch.bfloat16)]
        _,pos,mask,_,emb,packed=self.base.prepare_inputs_labels_for_multimodal(ids,None,self.torch.ones_like(ids),None,targets,visual(octa),visual(cfp))
        if emb.shape[1]>8192:raise ValueError('Do not truncate evidence/response supervision')
        hidden=self.base.get_model()(inputs_embeds=emb,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True).last_hidden_state
        device=self.base.lm_head.weight.device;labels=packed[:,1:].to(device);select=labels!=-100
        hidden=hidden[:,:-1].to(device)[select];labels=labels[select]
        logits=self.torch.nn.functional.linear(hidden,self.base.lm_head.weight).float()
        return logits.log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)

    def screening_score(self, row):
        """The exact first-token classifier used by the mixed-input trainer."""
        from retina2cardio.model_utils import encode_text
        from retina2cardio.prompts import SYSTEM
        from retina2cardio.model_utils import image_inputs
        from retina2cardio.model_utils import binary_ids
        rendered = self.tokenizer.apply_chat_template(
            [{'role':'system','content':SYSTEM}, {'role':'user','content':row['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.torch.tensor([encode_text(self.tokenizer,rendered)],device=self.device)
        cfp,octa = image_inputs(row['views'],self.root,self.base)
        visual = lambda x:[x.to('cuda:0',dtype=self.torch.bfloat16)]
        with self.torch.inference_mode():
            _,pos,mask,_,emb,_ = self.base.prepare_inputs_labels_for_multimodal(
                ids,None,self.torch.ones_like(ids),None,None,visual(octa),visual(cfp))
            hidden = self.base.get_model()(inputs_embeds=emb,attention_mask=mask,position_ids=pos,
                use_cache=False,return_dict=True).last_hidden_state[:,-1]
            device = self.base.lm_head.weight.device
            logits = self.torch.nn.functional.linear(hidden.to(device).float(),
                self.base.lm_head.weight[binary_ids(self.tokenizer)].float())
            return float(logits.softmax(-1)[0,1].cpu())

    def raw_report_baseline(self, row, probability, max_new_tokens=1000):
        """Unedited model continuation conditioned on its own class prediction."""
        from retina2cardio.model_utils import encode_text
        from retina2cardio.prompts import SYSTEM
        from retina2cardio.model_utils import image_inputs
        rendered = self.tokenizer.apply_chat_template(
            [{'role':'system','content':SYSTEM}, {'role':'user','content':row['prompt']}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prediction = int(probability >= .5)
        prefix = encode_text(self.tokenizer,rendered) + self.tokenizer.encode(str(prediction)+'\n',add_special_tokens=False)
        ids = self.torch.tensor([prefix],device=self.device)
        cfp,octa = image_inputs(row['views'],self.root,self.base)
        visual = lambda x:[x.to('cuda:0',dtype=self.torch.bfloat16)]
        with self.torch.inference_mode():
            tokens = self.base.generate(inputs=ids,attention_mask=self.torch.ones_like(ids),
                images=visual(cfp),octs=visual(octa),do_sample=False,max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
        return dict(model_generated_text=self.tokenizer.decode(tokens[0],skip_special_tokens=True),
                    prediction_prefill=str(prediction)+'\n',
                    response_token_count=int(tokens.shape[-1]),hit_token_limit=tokens.shape[-1]>=max_new_tokens)

    def __call__(self,stage,row,state):
        from retina2cardio.model_utils import encode_text
        from retina2cardio.model_utils import image_inputs
        prompt=policy_prompt(stage,row,state)
        rendered=self.tokenizer.apply_chat_template([{'role':'system','content':'You are a retinal imaging research assistant. Follow the JSON tool protocol; do not invent measurements.'},
            {'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.torch.tensor([encode_text(self.tokenizer,rendered)],device=self.device)
        if ids.shape[1]+len(row['views'])*(self.base.get_vision_tower().num_patches+1)+1600>8192:
            raise ValueError('No silent truncation of supplied images/tool evidence')
        cfp,octa=image_inputs(row['views'],self.root,self.base)
        visual=lambda x:[x.to('cuda:0',dtype=self.torch.bfloat16)]
        kw=dict(do_sample=self.sampling,max_new_tokens=700 if stage=='plan' else 1600,use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        if self.sampling:kw.update(temperature=.8,top_p=.95)
        was_training=self.model.training
        self.model.eval()
        try:
            with self.torch.inference_mode():
                tokens=self.base.generate(inputs=ids,attention_mask=self.torch.ones_like(ids),images=visual(cfp),octs=visual(octa),**kw)
        finally:
            self.model.train(was_training);self.base.get_vision_tower().eval();self.base.get_oct_2d_tower().eval()
        self.last_response_ids=tokens[0].detach().cpu().tolist()
        eos=self.base.generation_config.eos_token_id
        ends={eos} if isinstance(eos,int) else set(eos or [])
        self.last_response_truncated=not any(t in ends for t in self.last_response_ids)
        return self.tokenizer.decode(tokens[0],skip_special_tokens=True)
