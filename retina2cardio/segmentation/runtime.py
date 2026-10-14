"""Native Qwen2.5-VL + frozen SAM2, with response-token-only policy updates."""
from pathlib import Path

import numpy as np
from PIL import Image

from .rewards import SIZE, gold_box

PROMPT_CONTRACT = "paper_appendix_A1_target_specific_mask_acceptance_v47"


def mask_acceptance_instruction(query):
    target = query.strip().lower()
    if 'faz' in target or 'foveal avascular' in target:
        return ('Accept only when the visible foveal avascular zone boundary is represented. '
                'The FAZ is a region, not a thin branching vessel network. '
                'Do not infer physical area or disease normality from mask pixels.')
    if 'optic disc' in target or 'optic_disc' in target:
        return ('Accept only when the visible optic-disc region and boundary are represented. '
                'The disc is a region, not a thin branching vessel network. '
                'Do not infer physical diameter or disease normality from mask pixels.')
    if 'vessel' in target:
        return ('A solid green retinal field does not outline individual blood vessels. '
                'Accept only when the visible thin branching vessel network is segmented.')
    raise ValueError('Unregistered retinal segmentation target in query')


def action_instruction(task, query):
    """Appendix A.1 full tagged examples; identical protocol in train and eval."""
    if task == "TG":
        instruction = (f'Let\'s find out "{query}" with bbox step by step. '
            'Output brief visual reasoning in <think> </think> and the final answer in <answer> </answer>. '
            'Output the target bbox in JSON format. Here is the exact output template: '
            '<think>visual reasoning here</think><answer>{"bbox_2d": [x1, y1, x2, y2]}</answer>. '
            'Replace x1,y1,x2,y2 with integers on this 840x840 image; 0 <= x1 < x2 <= 840 '
            'and 0 <= y1 < y2 <= 840. ')
    elif task in ("MC", "AUX"):
        instruction = (f'Point out errors in the green segmentation mask of "{query}". '
            'Output brief visual reasoning in <think> </think> and the final answer in <answer> </answer>. '
            'Give the center of a missed target area as pos_point and an extra foreground area as neg_point. '
            'Use null when no such error exists; two nulls accept the mask. Exact output template: '
            '<think>visual reasoning here</think><answer>{"pos_point": null, "neg_point": [x, y]}</answer>. '
            'Always include BOTH pos_point AND neg_point keys. An unused key must be present with value null. '
            'If both points are used, they must be different pixels. '
            'Replace x,y with integer pixel coordinates 0..839 on this 840x840 image. ')
    else:
        raise ValueError("Unknown segmentation task")
    return instruction + ('The answer contains exactly one JSON object, without a code block, outer list, '
                          'extra keys, or trailing comma. Always finish with </answer>. '
                          'Describe visible retinal structures only.')




def bind_image(tool,record,image):
    if getattr(tool,'source_binding_required',False) is True:
        target = record.get('segmentation_target')
        if target is None and record['query'].strip().lower() in (
                'all visible retinal blood vessels','retinal blood vessels','retinal vessels'):
            target = 'retinal_vessels'
        source = {k:record[k] for k in ('image','image_sha256','modality')}
        source['segmentation_target'] = target
        tool.bind_record(source)
    tool.set_image(image)


def create_tool(args,cache_root):
    profile = getattr(args,'tool_profile','cfp_crop_recheck')
    if profile == 'cfp_crop_recheck' and getattr(args,'cfp_runtime',None):
        from .specialist_runtime import FrozenCFPCropTool
        return FrozenCFPCropTool(args.cfp_runtime,cache_root)
    if profile == 'cfp_disc_crop_recheck' and getattr(args,'cfp_runtime',None):
        from .specialist_runtime import FrozenCFPCropTool
        return FrozenCFPCropTool(args.cfp_runtime,cache_root,target='optic_disc')
    if profile == 'octa_native_sam2' and getattr(args,'octa_runtime',None):
        from .sam2_octa_tool import FrozenOCTASAM2
        return FrozenOCTASAM2(args.sam_checkpoint,args.octa_runtime)
    if profile == 'octa_faz_native_sam2' and getattr(args,'octa_runtime',None):
        from .sam2_octa_tool import FrozenOCTASAM2
        return FrozenOCTASAM2(args.sam_checkpoint,args.octa_runtime,target='faz')
    raise ValueError('A verified modality/target specialist is required; generic SAM2 remains diagnostic-only')


def overlay(image, mask):
    array = np.asarray(image).astype(float).copy()
    array[mask] = .55*array[mask]+.45*np.array([0., 255., 0.])
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def corrupt(gold, rng):
    """Independent p=.5 polygon inclusion/exclusion, appendix A.4.

    Polygon count, size distribution and rasterization are not specified in the
    paper. These are explicit local choices, not asserted author-code parity.
    """
    import cv2
    value = gold.copy()
    for add in (True, False):
        if rng.random() < .5:
            cx, cy = rng.integers(60, SIZE-60, size=2)
            angles = np.sort(rng.uniform(0, 2*np.pi, 6))
            radii = rng.uniform(30, 110, 6)
            pts = np.stack([cx+radii*np.cos(angles), cy+radii*np.sin(angles)], -1)
            polygon = np.zeros(gold.shape, np.uint8)
            cv2.fillPoly(polygon, [np.clip(pts, 0, SIZE-1).astype(np.int32)], 1)
            value[polygon.astype(bool)] = add
    return value


class FrozenSAM:
    def __init__(self, checkpoint, config="configs/sam2/sam2_hiera_l.yaml"):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        model = build_sam2(config, checkpoint, device="cuda", apply_postprocessing=False).eval()
        model.requires_grad_(False)
        self.predictor = SAM2ImagePredictor(model)
        self.initial_box = self.initial_logits = self.current = None
        self.provenance = dict(backend='official_frozen_SAM2_large',
                               checkpoint=str(checkpoint),model_is_Meta_SAM=True)

    def set_image(self, image):
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            self.predictor.set_image(np.array(image, copy=True))
        self.initial_box = self.initial_logits = self.current = None

    def _predict(self, box, points=None, labels=None, logits=None):
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            masks, scores, low = self.predictor.predict(
                box=np.asarray(box, np.float32),
                point_coords=None if not points else np.asarray(points, np.float32),
                point_labels=None if not labels else np.asarray(labels, np.int32),
                mask_input=logits, multimask_output=True)
        if not np.isfinite(scores).all() or not np.isfinite(low).all():
            raise ValueError("Nonfinite actual SAM output")
        # No oracle best-of-three by gold. Use SAM's predicted quality only.
        chosen = int(np.argmax(scores))
        return masks[chosen].astype(bool), low[chosen:chosen+1].copy()

    def prepare_mc(self, gold):
        return self.initialize_box(gold_box(gold))

    def initialize_box(self,box):
        self.initial_box = list(box)
        self.current, self.initial_logits = self._predict(self.initial_box)
        return self.current.copy()

    def step(self,action):
        points, labels = [], []
        for key,label in (('pos_point',1),('neg_point',0)):
            if action[key] is not None:
                points.append(action[key])
                labels.append(label)
        if points:
            self.current,self.initial_logits = self._predict(self.initial_box,points,labels,self.initial_logits.copy())
        return self.current.copy()

    def __call__(self, action):
        if "bbox_2d" in action:
            return self._predict(action["bbox_2d"])[0]
        if self.initial_logits is None:
            raise RuntimeError("MC must have an actual SAM initial state")
        points, labels = [], []
        for key, label in (("pos_point", 1), ("neg_point", 0)):
            if action[key] is not None:
                points.append(action[key])
                labels.append(label)
        if not points:
            return self.current.copy()
        # EVERY training rollout starts from the SAME state. Never accumulate
        # points from other rollouts, and retain the initial box as in the paper.
        return self._predict(self.initial_box, points, labels, self.initial_logits.copy())[0]


def response_prefix(tokens, eos):
    """Keep the first EOS, exclude padding; absence of EOS is not completion."""
    ends = {eos} if isinstance(eos, int) else set(eos or [])
    for index, token in enumerate(tokens):
        if int(token) in ends:
            return tokens[:index+1], True
    return tokens, False


class NativeActor:
    def __init__(self, checkpoint, max_response=1300, rank=16, sampling_micro_batch=1, adapter=None):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        if not Path(checkpoint).is_dir():
            raise ValueError("A local native Qwen2.5-VL checkpoint is required")
        self.torch, self.max_response = torch, max_response
        if type(sampling_micro_batch) is not int or not 1 <= sampling_micro_batch <= 4:
            raise ValueError('Sampling micro-batch must be an integer in [1,4]')
        self.sampling_micro_batch = sampling_micro_batch
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True,
                                                       min_pixels=SIZE*SIZE, max_pixels=SIZE*SIZE)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={"": 0}, attn_implementation="sdpa")
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={"use_reentrant": False})
        # Avoid adapting the vision encoder and avoid a second 7B reference copy.
        targets = [name for name, _ in base.named_modules()
                   if "language_model.layers." in name and name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
        if not targets:
            raise ValueError("Native language attention modules not found")
        self.model = (PeftModel.from_pretrained(base, adapter, is_trainable=True)
            if adapter else get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank*2,
                lora_dropout=0., target_modules=targets, task_type="CAUSAL_LM", bias="none")))
        self.model.config.use_cache = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        self.model.eval()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("No policy parameters are trainable")

    def inputs(self, image, task, query, current=None):
        if task == "TG":
            visible = image
        else:
            if current is None:
                raise ValueError("The actual mask overlay is required")
            visible = overlay(image, current)
        instruction = action_instruction(task, query)
        if current is not None:
            instruction += (f' The actual returned mask covers {100*float(current.mean()):.1f}% of image pixels. '
                            + mask_acceptance_instruction(query))
        messages = [{"role": "user", "content": [{"type": "image", "image": visible},
                                                       {"type": "text", "text": instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[visible], return_tensors="pt").to("cuda")
        if inputs.input_ids.shape[1] > 1300:
            raise ValueError("Prompt exceeds the paper's 1300-token limit; no silent truncation")
        return inputs

    def sample(self, inputs, count):
        self.model.eval()
        items = []
        with self.torch.inference_mode():
            for offset in range(0, count, self.sampling_micro_batch):
                micro = min(self.sampling_micro_batch, count-offset)
                # Qwen expands packed visual tensors using its native geometry.
                ids = self.model.generate(**inputs, do_sample=True, temperature=1., top_p=1., top_k=0,
                    num_return_sequences=micro, max_new_tokens=self.max_response,
                    repetition_penalty=1., use_cache=True)
                if len(ids) != micro:
                    raise ValueError('Native generator returned an incomplete micro-batch')
                eos = self.model.generation_config.eos_token_id
                for row in ids:
                    response, ended = response_prefix(row[inputs.input_ids.shape[1]:].detach().cpu(), eos)
                    text = self.processor.tokenizer.decode(response, skip_special_tokens=True)
                    items.append({"text": text, "ids": response,
                        "truncated": len(response) == self.max_response and not ended})
        return items

    def logp(self, inputs, response):
        torch = self.torch
        # Training mode activates gradient checkpointing; dropout is disabled
        # explicitly so old/reference/current probabilities stay comparable.
        self.model.train()
        tokens = response[None].to("cuda")
        data = {k: v for k, v in inputs.items()}
        data["input_ids"] = torch.cat([inputs["input_ids"], tokens], dim=1)
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        # Reset generation's cached RoPE delta by a fresh full-sequence forward.
        logits = self.model(**data, use_cache=False, logits_to_keep=len(response)+1).logits
        logits = logits[:, -len(response)-1:-1, :].float()
        return logits.log_softmax(-1).gather(-1, tokens[..., None]).squeeze(0).squeeze(-1)

    def old_and_ref(self, inputs, response):
        with self.torch.no_grad():
            old = self.logp(inputs, response).detach().cpu()
            with self.model.disable_adapter():
                ref = self.logp(inputs, response).detach().cpu()
        return old, ref
