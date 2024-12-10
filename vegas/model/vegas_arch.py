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

"""
modified from video-llava
"""

from abc import ABC, abstractmethod
import torch
from .multimodal_encoder.builder import build_image_tower, build_video_tower, build_audio_tower
from .multimodal_projector.builder import build_vision_projector
from vegas.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX,AUDIO_TOKEN_INDEX, SUB_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if getattr(config, "mm_image_tower", None) is not None:
            self.image_tower = build_image_tower(config, delay_load=True)
        if getattr(config, "mm_video_tower", None) is not None:
            self.video_tower = build_video_tower(config, delay_load=True)
            # self.video_tower = build_video_tower(config, delay_load=False)
        if getattr(config, "mm_audio_tower", None) is not None:
            self.audio_tower = build_audio_tower(config, delay_load=True)
        if getattr(config, "mm_image_tower", None) is not None or getattr(config, "mm_video_tower",
                                                                          None) is not None or getattr(config,
                                                                                                       "mm_audio_tower",
                                                                                                       None) is not None:
            self.mm_projector = build_vision_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        image_tower = model_args.image_tower
        video_tower = model_args.video_tower
        audio_tower = model_args.audio_tower
        assert image_tower is not None or video_tower is not None or audio_tower is not None
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        mm_audio_select_layer = model_args.mm_audio_select_layer
        mm_audio_select_feature = model_args.mm_audio_select_feature

        if isinstance(mm_audio_select_layer, str):
            try:
                mm_audio_select_layer = int(mm_audio_select_layer)
            except ValueError:
                mm_audio_select_layer = -1

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            print("Initializing image_tower")
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    image_tower = self.image_tower[0]
                else:
                    image_tower = self.image_tower
                image_tower.load_model()

        self.config.mm_video_tower = video_tower
        if video_tower is not None:
            if self.get_video_tower() is None:
                video_tower = build_video_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [video_tower]
                else:
                    self.video_tower = video_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    video_tower = self.video_tower[0]
                else:
                    video_tower = self.video_tower
                video_tower.load_model()

        self.config.mm_audio_tower = audio_tower
        if audio_tower is not None:
            if self.get_audio_tower() is None:
                audio_tower = build_audio_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.audio_tower = [audio_tower]
                else:
                    self.audio_tower = audio_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    audio_tower = self.audio_tower[0]
                else:
                    audio_tower = self.audio_tower
                audio_tower.load_model()


        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.config.mm_audio_select_layer = mm_audio_select_layer
        self.config.mm_audio_select_feature = mm_audio_select_feature

        if image_tower is not None and video_tower is not None and audio_tower is None:  # TODO: support different hidden_size
            assert image_tower.hidden_size == video_tower.hidden_size
            self.config.mm_hidden_size = image_tower.hidden_size
        elif image_tower is not None and video_tower is not None and audio_tower is not None:  # TODO: support different hidden_size
            assert image_tower.hidden_size == video_tower.hidden_size == audio_tower.hidden_size
            self.config.mm_hidden_size = image_tower.hidden_size
        else:
            self.config.mm_hidden_size = max(getattr(image_tower, 'hidden_size', -1),
                                             getattr(video_tower, 'hidden_size', -1),
                                             getattr(audio_tower, 'hidden_size', -1))

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_videos(self, videos, text_querys=None):  # [mini_b, c, t, h, w]
        b, _, t, _, _ = videos.shape
        video_features,_= self.get_model().get_video_tower()(videos, text_querys)
        video_features = self.get_model().mm_projector(video_features)
        return video_features

    def encode_audios(self, audios, text_querys = None):  # [mini_b, c, t, h, w]
        b, _, _, _ = audios.shape
        audio_features,_ = self.get_model().get_audio_tower()(audios, text_querys)  # [mini_b, 3, bins, len]
        audio_features = self.get_model().mm_projector(audio_features)
        return audio_features


    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images, text_querys, sub_strings=None, input_ids_sub=None, subid_attention_mask=None ):
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()
        audio_tower = self.get_audio_tower()
        if (image_tower is None and video_tower is None and audio_tower is None) or images is None or input_ids.shape[
            1] == 1  or len(images) == 0:
            if past_key_values is not None and (
                    image_tower is not None or video_tower is not None or audio_tower is not None) and images is not None and \
                    input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_idx = [idx for idx, img in enumerate(images) if isinstance(img, torch.Tensor) and img.ndim == 3]
        video_idx = [idx for idx, vid in enumerate(images) if isinstance(vid, torch.Tensor) and vid.ndim == 4]
        audio_idx = [idx for idx, aud in enumerate(images) if isinstance(aud, torch.Tensor) and aud.ndim == 5]
        subtitle_idx = [idx for idx, sub in enumerate(images) if isinstance(sub, str)]
        # image_idx = []

        if subtitle_idx:
            if subid_attention_mask is None:
                subid_attention_mask = torch.ones_like(input_ids_sub, dtype=torch.bool, device=input_ids_sub.device)
            else:
                subid_attention_mask = subid_attention_mask.bool().to(input_ids_sub.device)
            sub_strings = [sub_string for sub_string in sub_strings if sub_string != '']

            real_sub_ids = [idx for idx, sub_string in enumerate(sub_strings) if sub_string]

            try:
                input_ids_sub = torch.stack([input_ids_sub[idx] for idx in real_sub_ids])
                subid_attention_mask = torch.stack([subid_attention_mask[idx] for idx in real_sub_ids]).to(input_ids_sub.device)
            except:
                print('has empty in input_ids_sub, skip this sample with subtitle')
                bs = input_ids_sub.shape[0]
                for idx in real_sub_ids:
                    input_ids_sub[idx] = torch.zeros(bs, 1, dtype=torch.long).to(input_ids_sub.device)
                if subid_attention_mask is None:
                    subid_attention_mask = torch.ones_like(input_ids_sub, dtype=torch.bool, device=input_ids_sub.device)
                else:
                    subid_attention_mask = subid_attention_mask.bool().to(input_ids_sub.device)

        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []  # mini_b c h w
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []  # mini_b c t h w
        audios_minibatch = torch.stack([images[idx].squeeze(0).squeeze(-1) for idx in audio_idx]) if len(audio_idx) > 0 else []

        tmp_image_features = [None] * (len(image_idx) + len(video_idx) + len(audio_idx) + len(subtitle_idx))
        if getattr(images_minibatch, 'ndim', 0) == 4:
            if image_tower is not None:
                image_features_minibatch = self.encode_images(images_minibatch)
            else:
                image_features_minibatch = torch.randn(1).to(self.device)

            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

        to_replace = ["<video>\n", "<audio>\n", "<sub>\n", "<video>", "<audio>",
                      "Answer the question with a few words:", "Select the right answer from"]

        if getattr(videos_minibatch, 'ndim', 0) == 5:
            vid_text_querys = text_querys
            if text_querys:
                vid_text_querys = [text_query for text_query in text_querys if "<video>" in text_query]
                vid_text_querys = [query.replace("<video>", "") for query in vid_text_querys]

                for replacement in to_replace:
                    vid_text_querys = [query.replace(replacement, '') for query in vid_text_querys]
                assert len(vid_text_querys) == videos_minibatch.shape[0], "Unequal number of text_querys and videos_minibatch"
            video_features_minibatch, indices = self.encode_videos(videos_minibatch,text_querys=vid_text_querys)

            for i, pos in enumerate(video_idx):
                t = video_features_minibatch[i].shape[0]
                tmp_image_features[pos] = [video_features_minibatch[i][j] for j in range(t)]

        if getattr(audios_minibatch, 'ndim', 0) == 4:
            if audio_tower is not None:
                aud_text_querys = None
                audio_features_minibatch = self.encode_audios(audios_minibatch, text_querys=aud_text_querys)
            else:
                audio_features_minibatch = torch.randn(1).to(self.device)

            for i, pos in enumerate(audio_idx):
                if pos >= len(tmp_image_features):
                    continue
                if i >= len(audio_features_minibatch):
                    print('i >= len(audio_features_minibatch)')
                    tmp_image_features[pos] = audio_features_minibatch[-1]
                else:
                    tmp_image_features[pos] = audio_features_minibatch[i]

        if subtitle_idx:
            input_ids_sub = [cur_sub_ids[cur_attention_mask] for cur_sub_ids, cur_attention_mask in
                             zip(input_ids_sub, subid_attention_mask)]

            bs_sub_embeds = []
            for sub_id in input_ids_sub:
                sub_embed = self.get_model().embed_tokens(sub_id)
                bs_sub_embeds.append(sub_embed)

            for i, pos in enumerate(subtitle_idx):
                if pos >= len(tmp_image_features):
                    continue
                if i >= len(bs_sub_embeds):
                    tmp_image_features[pos] = bs_sub_embeds[-1]
                else:
                    tmp_image_features[pos] = bs_sub_embeds[i]

        new_tmp = []
        for image in tmp_image_features:
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
            else:
                new_tmp.append(image)

        image_features = new_tmp

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() + (cur_input_ids == AUDIO_TOKEN_INDEX).sum() + (cur_input_ids == SUB_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]],
                                             dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where((cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == AUDIO_TOKEN_INDEX) | (cur_input_ids == SUB_TOKEN_INDEX))[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            truncated_samples_indices = [i for i, x in enumerate(new_input_embeds) if len(x) > tokenizer_model_max_length]
            truncated_samples = len(truncated_samples_indices)
            if truncated_samples > 0:
                print(f"{truncated_samples} samples will be truncated because their length exceeded {tokenizer_model_max_length}.")

            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels



    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
