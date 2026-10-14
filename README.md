# Retina2Cardio-VLM


## Install


```bash
git clone https://github.com/lixiangcog/Retina2Cardio-VLM.git
cd Retina2Cardio-VLM
pip install -e ".[model,agent,grpo,screening]"
```


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
