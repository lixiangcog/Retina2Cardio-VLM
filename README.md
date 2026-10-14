# Retina2Cardio-VLM

Report-guided visual evidence acquisition for cardiovascular screening research
from CFP and OCTA. **Method code only:** no datasets, experiment scripts, results,
private reports, credentials, or weights are included.

## Method

`Imaging question → select source and ROI → segment and refine → measure → revise report`

- **Reporter:** unquantized BF16 Qwen3-8B with external CFP/OCTA RETFound branches.
- **Visual agent:** SAM-Veteran-inspired Qwen2.5-VL controller with QLoRA GRPO,
  actual mask feedback, correction and explicit stopping.
- **Specialists:** RetSAM for CFP vessels/optic disc; target-adapted SAM2 for OCTA
  vessels/FAZ. CFP crop rechecking is not native Meta SAM point prompting.
- **Learning:** segmentation-policy GRPO, report trajectory SFT and separate
  report/tool-selection GRPO. Frozen-feature fusion screening is also provided.

Inputs are ordered `CFP, DCP, SCP, RPCP` or `CFP, OCTA_UNKNOWN`. Missing layers
are not inferred. OCT is not yet supported.

## Install

Install a compatible CUDA PyTorch/torchvision pair, then:

```bash
git clone https://github.com/lixiangcog/Retina2Cardio-VLM.git
cd Retina2Cardio-VLM
pip install -e ".[model,agent,grpo,screening]"
```

Install [SAM2](https://github.com/facebookresearch/sam2) separately. Obtain Qwen,
RETFound, RetSAM and task-specific adapter weights locally under their licenses.

## Run

```bash
python -m retina2cardio \
  --request /local/private/request.json \
  --image-root /local/private/images \
  --checkpoint /local/weights/report_policy \
  --registry /local/private/specialists.json \
  --output /local/private/report_output \
  --report-gpus 0,1 --tool-gpu 2
```

- Request fields: `id`, `views`, `clinical`, `biomarker_evidence`, `prompt`.
  Each view contains `path`, `modality`, `role`, `sha256`; use relative image paths
  and actual SHA256 values. Never supply true CAD labels or teacher answers.
- Reporter checkpoint: LoRA adapter, `non_lora_trainables.bin`, and
  `cad_run_config.json` containing `model_base`, `vision_weights`, `quantized: false`.
- Registry: `tools` entries keyed by modality/target, with `profile`, `actor`,
  `policy_checkpoint`, specialist runtime, `promotion_receipt`, and `source_qc`.
  Actual checkpoint, validation and source hashes are checked by `tools.py`.
- Outputs preserve policy turns, real tool masks/actions, measurement receipts
  and the report in `episode.json`. Rejected measurements remain unavailable.

Training APIs are `training.segmentation_grpo_step`, `training.report_sft_step`,
and `training.report_grpo_step`. Callers supply private reviewed trajectories,
independent segmentation annotations, optimizers and checkpoints. No dataset
construction or experiment loops are bundled.

## Limits and licenses

This is a research implementation, not a clinically validated diagnostic device
or an author-code-identical SAM-Veteran reproduction. Geometry outputs are pixel
proxies, not calibrated clinical vessel density, caliber or FAZ area. FAZ is
restricted to anatomically reviewed SCP/DCP sources, not RPCP/unknown layers.
Valid target checkpoints and source QC are required; do not bypass these gates.

Inspired by [GROK](https://github.com/lixiangcog/GROK-A-Retinal-MLLMs) and
[SAM-Veteran](https://proceedings.iclr.cc/paper_files/paper/2026/file/eb115a5de129c1b34c30dc5ee72c2ec7-Paper-Conference.pdf).
Original code is MIT; LLaVA-derived code is Apache-2.0 and RETFound-derived code
is **CC BY-NC 4.0**. See `NOTICE` and `LICENSES`; external weights retain their
own terms.
