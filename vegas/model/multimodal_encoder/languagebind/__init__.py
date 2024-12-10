import torch
from torch import nn
from transformers import AutoConfig

from .image.configuration_image import LanguageBindImageConfig
from .image.modeling_image import LanguageBindImage
from .image.tokenization_image import LanguageBindImageTokenizer
from .image.processing_image import LanguageBindImageProcessor

from .video.configuration_video import LanguageBindVideoConfig
from .video.modeling_video import LanguageBindVideo
from .video.tokenization_video import LanguageBindVideoTokenizer
from .video.processing_video import LanguageBindVideoProcessor

from .video.crema_topk import TopK_Selector
# from .video.crema_linear_topk import TopK_Selector

from .depth.configuration_depth import LanguageBindDepthConfig
from .depth.modeling_depth import LanguageBindDepth
from .depth.tokenization_depth import LanguageBindDepthTokenizer
from .depth.processing_depth import LanguageBindDepthProcessor

from .audio.configuration_audio import LanguageBindAudioConfig
from .audio.modeling_audio import LanguageBindAudio
from .audio.tokenization_audio import LanguageBindAudioTokenizer
from .audio.processing_audio import LanguageBindAudioProcessor

from .thermal.configuration_thermal import LanguageBindThermalConfig
from .thermal.modeling_thermal import LanguageBindThermal
from .thermal.tokenization_thermal import LanguageBindThermalTokenizer
from .thermal.processing_thermal import LanguageBindThermalProcessor

import torch
from transformers import AutoTokenizer, CLIPTextModel
from einops import rearrange
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPAttention




config_dict = {
    'thermal': LanguageBindThermalConfig,
    'image': LanguageBindImageConfig,
    'video': LanguageBindVideoConfig,
    'depth': LanguageBindDepthConfig,
    'audio': LanguageBindAudioConfig
}
model_dict = {
    'thermal': LanguageBindThermal,
    'image': LanguageBindImage,
    'video': LanguageBindVideo,
    'depth': LanguageBindDepth,
    'audio': LanguageBindAudio
}
transform_dict = {
    'video': LanguageBindVideoProcessor,
    'audio': LanguageBindAudioProcessor,
    'depth': LanguageBindDepthProcessor,
    'thermal': LanguageBindThermalProcessor,
    'image': LanguageBindImageProcessor,
}


class LanguageBind(nn.Module):
    def __init__(self, clip_type=('thermal', 'image', 'video', 'depth', 'audio'), use_temp=True,
                 cache_dir='./cache_dir'):
        super(LanguageBind, self).__init__()
        self.use_temp = use_temp
        self.modality_encoder = {}
        self.modality_proj = {}
        self.modality_scale = {}
        self.modality_config = {}
        for c in clip_type:
            pretrained_ckpt = f'LanguageBind/LanguageBind_{c.capitalize()}'
            model = model_dict[c].from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
            self.modality_encoder[c] = model.vision_model
            self.modality_proj[c] = model.visual_projection
            self.modality_scale[c] = model.logit_scale
            self.modality_config[c] = model.config
        self.modality_encoder['language'] = model.text_model
        self.modality_proj['language'] = model.text_projection

        self.modality_encoder = nn.ModuleDict(self.modality_encoder)
        self.modality_proj = nn.ModuleDict(self.modality_proj)

    def forward(self, inputs):
        outputs = {}
        for key, value in inputs.items():
            value = self.modality_encoder[key](**value)[1]
            value = self.modality_proj[key](value)
            value = value / value.norm(p=2, dim=-1, keepdim=True)
            if self.use_temp:
                if key != 'language':
                    value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs


def to_device(x, device):
    out_dict = {k: v.to(device) for k, v in x.items()}
    return out_dict


class LanguageBindImageTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()
        self.is_loaded = False
        self.image_tower_name = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindImageConfig.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindImage.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower = model.vision_model
        self.image_tower.requires_grad_(False)
        self.image_processor = LanguageBindImageProcessor(model.config)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                     output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype),
                                                  output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.embeddings.class_embedding.dtype  #############

    @property
    def device(self):
        return self.image_tower.embeddings.class_embedding.device  ##############

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

import json
from types import SimpleNamespace
class LanguageBindVideoTower(nn.Module):
    def __init__(self, video_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()
        self.is_loaded = False
        self.num_frames = 8
        self.video_tower_name = video_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.tokenizer = AutoTokenizer.from_pretrained("root/my_cache_dir/clip-vit-base-patch32/")
        self.sampler = True

        print('use sample? ', self.sampler)
        try:
            self.train_sampler = args.train_sampler
        except:
            self.train_sampler = False

        if self.sampler:
            self.sampler = TopK_Selector(num_select=self.num_frames)

        self.extra_tempo_embed = True
        if self.extra_tempo_embed:
            self.v_hidden_size = 1024
            self.extra_temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, self.v_hidden_size))
            nn.init.normal_(self.extra_temporal_embedding, std=self.v_hidden_size ** -0.5)

            video_bind_cfg = '/cache_dir/LanguageBind_Video_merge/config.json'
            with open(video_bind_cfg, "r") as f:
                config_data = json.load(f)
            extra_config = SimpleNamespace(**config_data["vision_config"])
            self.time_att_extra = CLIPAttention(extra_config)
            self.temporal_layer_norm1 = nn.LayerNorm(extra_config.hidden_size, eps=extra_config.layer_norm_eps)


        self.cache_dir = cache_dir
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindVideoConfig.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindVideo.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)
        self.video_processor = LanguageBindVideoProcessor(model.config)
        self.video_tower = model.vision_model
        self.video_tower.requires_grad_(False)
        self.text_tower = model.text_model
        self.text_tower.requires_grad_(False)

        if self.training and self.sampler and self.train_sampler:
            self.sampler.requires_grad_(True)
        if self.extra_tempo_embed and self.train_sampler:
            self.extra_temporal_embedding.requires_grad_(True)
            self.time_att_extra.requires_grad_(True)


        self.is_loaded = True

    def feature_select(self, video_forward_outs):
        video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        return video_features  # return all

    # @torch.no_grad()
    def forward(self, videos, text_querys=None, context_length=77):
        with torch.no_grad():
            if text_querys and self.sampler:
                if isinstance(text_querys, list):
                    inputs = self.tokenizer(text_querys, max_length=context_length, padding='max_length',
                                            truncation=True, return_tensors='pt').to(self.device)
                    text_output = self.text_tower(**inputs)
                    text_embeds = text_output[0]
                else:
                    print('text_querys should always be a list if its not None')


            video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype),
                                                  output_hidden_states=True)
            sequence_pooled = video_forward_outs.pooler_output
            video_features = self.feature_select(video_forward_outs).to(videos.dtype)

        if text_querys and self.sampler:
            video_features_t, indices = self.sampler(video_features, text_embeds, sequence_pooled)
            n = video_features_t.size(2)
            video_features_t = rearrange(video_features_t, 'b t n d -> (b n) t d', t=self.num_frames)

            video_features_t = video_features_t + self.extra_temporal_embedding[:, :self.num_frames, :]
            video_features_t = rearrange(video_features_t, '(b n) t d -> (b t) n d', n=n)

            residual = video_features_t
            video_features_t = rearrange(video_features_t, '(b t) n d -> (b n) t d', t=self.num_frames)
            video_features_t = self.temporal_layer_norm1(video_features_t)
            video_features_t, attn_weights = self.time_att_extra(hidden_states=video_features_t)
            video_features_t = residual + rearrange(video_features_t, '(b n) t d -> (b t) n d', n=n)
            video_features_t = rearrange(video_features_t, '(b t) n d -> b t n d',  t=self.num_frames)


            return video_features_t, sequence_pooled


        return video_features, sequence_pooled

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.video_tower.embeddings.class_embedding.dtype  #############

    @property
    def device(self):
        return self.video_tower.embeddings.class_embedding.device  ##############

    @property
    def config(self):
        if self.is_loaded:
            return self.video_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class LanguageBindAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()
        self.is_loaded = False
        self.audio_tower_name = audio_tower
        self.select_layer = -2
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')
        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindAudioConfig.from_pretrained(self.audio_tower_name, cache_dir=self.cache_dir)

        self.reweight = False


    ############################################################
    def load_model(self):
        model = LanguageBindAudio.from_pretrained(self.audio_tower_name, cache_dir=self.cache_dir)
        self.audio_processor = LanguageBindAudioProcessor(model.config)
        self.audio_tower = model.vision_model
        self.audio_tower.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained("root/my_cache_dir/clip-vit-base-patch32/")

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.hidden_states[self.select_layer]  # b t n c
        return audio_features


    @torch.no_grad()
    def forward(self, audios, text_querys=None):
        if type(audios) is list:
            audio_features = []
            for audio in audios:
                audio_forward_out = self.audio_tower(audio.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                     output_hidden_states=True)
                audio_feature = self.feature_select(audio_forward_out).to(audio.dtype)
                audio_features.append(audio_feature)

        else:
            audio_forward_outs = self.audio_tower(audios.to(device=self.device, dtype=self.dtype),
                                                  output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(audios.dtype)  # bs,667,1024

        return audio_features, None

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.embeddings.class_embedding.dtype  #############

    @property
    def device(self):
        return self.audio_tower.embeddings.class_embedding.device  ##############

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2