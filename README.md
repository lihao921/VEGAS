<h2 align="center"> VEGAS: Towards Visually Explainable and Grounded Artificial Social Intelligence</h2>

<div align="center">
<!--   <img src="https://github.com/lihao921/VEGAS/blob/main/media/motivation.png" width="100%"> -->
    <img src="https://github.com/lihao921/VEGAS/blob/main/media/structure.png" width="80%">  

</div>

 
# Data preparation
## Training Data
 <img src="https://github.com/lihao921/VEGAS/blob/main/media/learning.png" width="100%">
 
### LGS Datasets
- Download Video-ChatGPT data used for the baseline [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) fine-tuning .
- Download [Next-QA]( https://doc-doc.github.io/docs/nextqa.html) data and Open-ended [Annotations](https://doc-doc.github.io/docs/nextqa.html).
- Download [TVQA](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa_plus.html) dataset.
- Download our annotation files from [google drive link] to filter the above datasets for training.

### MCQ Dataset
- Download Social-IQ raw videos and annotations from [Social-IQ](http://multicomp.cs.cmu.edu/resources/social-iq/).
- Download our processed annotation file with subtitles from [google drive link].
- Download GPT-refined Social-IQ annotations from [google drive link].



We also provide the processed data at  [google drive link].


After downloading all of them, organize the data as follows.
```Shell
MEDIA
├── Video
│   ├── videochatgpt_tune
│   ├── videochatgpt_shuffled
│   ├── tvqa
│   ├── nextqa
│   ├── cmu_mosei
│   ├── social_iq
├── Image
│   ├──expression_in_wild
├── Audio
│   ├── tvqa
│   ├── ravdess
│   ├── audio_caps
│   ├── cmu_mosei
```

### Pre-trained Weights Preparation
To finetune based on VideoLLaVA model, we opt for a temporary checkpoint weights replacing to avoid complex code re-implementation.
1. Download [vicuna-7b-v1.5 checkpoints](https://huggingface.co/lmsys/vicuna-7b-v1.5).
2. Download the pre-trained [Video-LLaVA checkpoints](https://huggingface.co/LanguageBind/Video-LLaVA-7B).
3. Replace weights as follows.
    ```shell
    python replace_vicuna_weights.py
    ```
Please ensure that you comply with the license and copyright terms of the original model.
### Train LGS

```shell
sbatch scripts/train_sampler.sh
```

### Train STP&LLM
```shell
sbatch scripts/train_joint_openended.sh
```

Or downloaded the trained checkpoints from [google drive link].

## Inference for video
```shell
python inference_siq.py \
  --model_base models--LanguageBind--Video-LLaVA-7B/ \
  --model_path joint_finetune/trained_weights/  \
  --video vegas/server/examples/1.mp4 \
  --lora \
```

## Updating...
