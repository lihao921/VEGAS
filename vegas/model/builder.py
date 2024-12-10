#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from vegas.model import *
from vegas.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, \
    DEFAULT_AUDIO_PATCH_TOKEN, DEFAULT_AUD_START_TOKEN, DEFAULT_AUD_END_TOKEN


def check_loaded_parameters(model_keys, module_keys, check_param_name):
    loaded_keys = []
    not_loaded_keys = []

    for key in module_keys:
        if key in model_keys:
            loaded_keys.append(key)
        else:
            not_loaded_keys.append(key)


    print("Load successful parameters>>>>>>>>>>>>>")
    print(loaded_keys)
    if not loaded_keys:
        print("Re-trained module parameters provided, but load FAILED! Perhaps wrong key names")

    print("Load failed parameters>>>>>>>>>>>>>")
    print(not_loaded_keys)

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    sampler_path = None
    if 'sampler_path' in kwargs:
        sampler_path = kwargs.get('sampler_path', None)
        _ = kwargs.pop('sampler_path')

    sampled_projector_path = None
    if 'sampled_projector_path' in kwargs:
        sampled_projector_path = kwargs.get('sampled_projector_path', None)
        _ = kwargs.pop('sampled_projector_path')


    video_tower_path = None
    if 'video_tower_path' in kwargs:
        video_tower_path = kwargs.get('video_tower_path', None)
        _ = kwargs.pop('video_tower_path')

    print('sampler path:', sampler_path)
    print('video tower_path:', video_tower_path)
    print('sampled_projector_path:', sampled_projector_path)

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    print('You are using model_name: ', model_name.lower())
    if 'llava' in model_name.lower():
        print('branch 1')
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            print('branch 1.1')
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower().lower() and model_base is not None:
            print('branch 1.2')
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

            if sampled_projector_path is not None:
                print('in builder.py: using sampled_projector_path...')
                sampled_projector_bin = os.path.join(sampled_projector_path, 'sampled_projector.bin')
                mm_projector_bin = os.path.join(sampled_projector_path, 'mm_projector.bin')

                if os.path.exists(sampled_projector_bin):
                    selected_projector_bin = sampled_projector_bin
                    print("Selected projector bin file:", selected_projector_bin)
                elif os.path.exists(mm_projector_bin):
                    selected_projector_bin = mm_projector_bin
                    print("Selected projector bin file:", selected_projector_bin)
                else:
                    raise FileNotFoundError("Could not find a projector bin file.")

                projector_weights = torch.load(selected_projector_bin, map_location='cpu')
                projector_weights = {k.replace("base_model.model.",""): v.to(torch.float16) for k, v in projector_weights.items()}  # incase saved with 'base_model.model.'
                model.load_state_dict(projector_weights, strict=False)

                model_keys = model.state_dict().keys()

                check_loaded_parameters(model_keys, projector_weights.keys(), 'mm_projector')



            if video_tower_path is not None:
                print('in builder.py: using video_tower_path...')
                tower_weights = torch.load(os.path.join(video_tower_path, 'video_tower.bin'),
                                           map_location='cpu')
                tower_weights = {k.replace("base_model.model.",""): v.to(torch.float16) for k, v in tower_weights.items()}   # incase saved with 'base_model.model.'
                model.load_state_dict(tower_weights, strict=False)
                model_keys = model.state_dict().keys()
                check_loaded_parameters(model_keys, tower_weights.keys(), 'video_tower')

        elif model_base is not None:
            print('branch 1.3')
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            print('Non-lora, load video_tower.bin')
            sampler_weights = torch.load(os.path.join(model_path, 'video_tower.bin'), map_location='cpu')
            sampler_weights = {k: v.to(torch.float16) for k, v in sampler_weights.items()}
            model.load_state_dict(sampler_weights, strict=False)

            print('Non-lora, load mm_projector.bin')
            sampler_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            sampler_weights = {k: v.to(torch.float16) for k, v in sampler_weights.items()}
            model.load_state_dict(sampler_weights, strict=False)

        else:
            print('---------branch 3, Non-lora, model_base is None---------')
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                if sampler_path is not None:
                    print('in builder.py: using sampler re-trained model, additionally loading sampler weights...')
                    sampler_weights = torch.load(os.path.join(sampler_path, 'video_tower_sampler.bin'),
                                                 map_location='cpu')
                    sampler_weights = {k: v.to(torch.float16) for k, v in sampler_weights.items()}
                    # print(list(sampler_weights.keys()))
                    model.load_state_dict(sampler_weights, strict=True)
                if sampled_projector_path is not None:
                    print('in builder.py: using sampler re-trained projector, additionally loading projector weights...')

                    projector_path = os.path.join(sampled_projector_path, 'sampled_projector.bin')
                    if os.path.exists(projector_path):
                        print('projector_path', projector_path)
                        projector_weights = torch.load(projector_path, map_location='cpu')
                    else:
                        projector_path = os.path.join(sampled_projector_path, 'mm_projector.bin')
                        print('projector_path', projector_path)
                        projector_weights = torch.load(projector_path, map_location='cpu')

                    # projector_weights = {k: v.to(torch.float16) for k, v in projector_weights.items()}
                    projector_weights = {k.replace("base_model.model.",""): v.to(torch.float16) for k, v in projector_weights.items()}
                    model.load_state_dict(projector_weights, strict=False)

                    model_keys = model.state_dict().keys()
                    check_loaded_parameters(model_keys, projector_weights.keys(), 'mm_projector')

                   
                if video_tower_path is not None:
                    print('in builder.py: using video_tower_path...')
                    tower_weights = torch.load(os.path.join(video_tower_path, 'video_tower.bin'),
                                                   map_location='cpu')
                    tower_weights = {k.replace("base_model.model.",""): v.to(torch.float16) for k, v in tower_weights.items()}
                    model.load_state_dict(tower_weights, strict=False)

                    model_keys = model.state_dict().keys()
                    check_loaded_parameters(model_keys, tower_weights.keys(), 'video_tower')


    else:
        print('branch 2')
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    processor = {'image': None, 'video': None, 'audio': None}
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_AUD_START_TOKEN, DEFAULT_AUD_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if model.config.mm_image_tower is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=device, dtype=torch.float16)
            image_processor = image_tower.image_processor
            processor['image'] = image_processor

        if model.config.mm_video_tower is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=torch.float16)
            video_processor = video_tower.video_processor
            processor['video'] = video_processor
        if model.config.mm_audio_tower is not None:
            audio_tower = model.get_audio_tower()
            if not audio_tower.is_loaded:
                audio_tower.load_model()
            audio_tower.to(device=device, dtype=torch.float16)
            audio_processor = audio_tower.audio_processor
            processor['audio'] = audio_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len

