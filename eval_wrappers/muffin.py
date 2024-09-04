# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import timm
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation

from typing import List, Optional, Tuple, Union

import os

from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from torchvision import transforms

from torchscale.model.BEiT3 import BEiT3 as BEiT3_base
from torchscale.architecture.config import EncoderConfig

import sys
sys.path.append('./utils')
from muffin.model.muffin import Beit3LlavaLlamaModel


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class BEiT3(BEiT3_base):
    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
        return_all_hiddens=False,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
            return_all_hiddens=return_all_hiddens
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)

        # self.apply(self._init_weights) # no longer necessary since we only use the pre-trained ckpt
        # self.mim_head = nn.Linear(1024, 8192)
        self.num_img_patches = self.beit3.vision_embed.num_position_embeddings()
        self.hidden_size = args.encoder_embed_dim

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pixel_values, query_embed=None, encode_image=False, img_feat_layer=-1, attn_mask=None):
        assert (query_embed is not None) ^ encode_image
        B = pixel_values.size(0)
        dtype = self.beit3.vision_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype)
        token_embeddings = self.beit3.vision_embed(pixel_values)
        multiway_split_position = -1
        if query_embed is not None:
            query_embed = torch.stack([query_embed] * B)
            multiway_split_position = token_embeddings.size(1)
            token_embeddings = torch.cat([token_embeddings, query_embed], dim=1)

        outputs = self.beit3.encoder(
            src_tokens=None,
            token_embeddings=token_embeddings,
            multiway_split_position=multiway_split_position,
            return_all_hiddens=encode_image,
            attn_mask=attn_mask,
        )
        vision_hidden_states = outputs["encoder_out"]
        if query_embed is not None:
            vision_hidden_states = vision_hidden_states[:, self.num_img_patches:]
        if encode_image:
            vision_hidden_states = outputs['encoder_states'][img_feat_layer][:, 1:self.num_img_patches]
        return vision_hidden_states

@register_model
def beit3_large_patch16_224(pretrained=False, **kwargs):
    args = _get_large_config(img_size=224, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model

@register_model
def beit3_large_patch16_256(pretrained=False, **kwargs):
    args = _get_large_config(img_size=256, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model

@register_model
def beit3_large_patch16_336(pretrained=False, **kwargs):
    args = _get_large_config(img_size=336, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model

@register_model
def beit3_large_patch16_448(pretrained=False, **kwargs):
    args = _get_large_config(img_size=448, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model

@register_model
def beit3_large_patch16_672(pretrained=False, **kwargs):
    args = _get_large_config(img_size=672, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model

@register_model
def beit3_large_itc_patch16_224(pretrained=False, **kwargs):
    args = _get_large_config(img_size=224, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model



DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def construct_query_parameter(query_k, h_size, init_weights, init=True):
    query_data = torch.zeros(query_k, h_size)
    if init:
        trunc_normal_(query_data, std=.02)
    for idx in range(query_k):
        if init_weights[idx] is not None:
            query_data[idx] = init_weights[idx]
    query = torch.nn.Parameter(query_data)
    return query


def interpolate_beit3(model, new_model_name):
    target_size = new_model_name.split('_')[-1]
    state_dict = model.state_dict()

    # interpolate position embedding
    pos_embed_key = 'beit3.encoder.embed_positions.A.weight'
    pos_embed_checkpoint = state_dict[pos_embed_key]
    embedding_size = pos_embed_checkpoint.shape[-1]

    # being consistent with Fairseq, which starts from 2 for position embedding
    torchscale_model = True
    num_patches = model.beit3.vision_embed.num_patches
    num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches

    # height (== width) for the checkpoint position embedding
    orig_size = int(num_patches ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(target_size) // 16
    # class_token and dist_token are kept unchanged

    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        if torchscale_model:
            new_pos_embed = new_pos_embed.squeeze(0)
        state_dict[pos_embed_key] = new_pos_embed

    return state_dict

# The implementation code is modified from DeiT (https://github.com/facebookresearch/deit.git)
def load_model_and_may_interpolate(checkpoint_model, model):
    state_dict = model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):
        if pos_embed_key in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_embed_key]
            embedding_size = pos_embed_checkpoint.shape[-1]
            if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                # being consistent with Fairseq, which starts from 2 for position embedding
                torchscale_model = True
                num_patches = model.beit3.vision_embed.num_patches
                num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches
            else:
                torchscale_model = False
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                if torchscale_model:
                    extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                else:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                if torchscale_model:
                    new_pos_embed = new_pos_embed.squeeze(0)
                checkpoint_model[pos_embed_key] = new_pos_embed
    return checkpoint_model


def build_transform(is_train, randaug=True, input_size=224, interpolation='bicubic'):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(
                input_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        if randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


class Beit3LlavaConfig(PretrainedConfig):
    model_type = "beit3_llava"


class Beit3Model(PreTrainedModel):
    config_class = Beit3LlavaConfig

    def __init__(self, config, mm_vision_tower=None, mm_hidden_size=None):
        super().__init__(config)
        mm_vision_tower = config.mm_vision_tower
        self.vision_tower = timm.create_model(mm_vision_tower)
        self.mm_projector = nn.Linear(self.vision_tower.hidden_size, config.hidden_size)
        self.query = torch.nn.Parameter(torch.zeros(config.num_query, self.vision_tower.args.encoder_embed_dim))
        self.vision_config = lambda x: None

    def initialize_vision_modules(self, vision_tower, no_randaug, num_query):
        self.config.mm_vision_tower = vision_tower
        self.config.use_mm_proj = True
        self.config.num_query = num_query

        if not hasattr(self, 'vision_tower'):
            vision_tower = timm.create_model(vision_tower)
            state_dict = torch.load('/mnt/data/user/tc_agi/multi_modal/checkpoints/beit-v3/beit3_large_patch16_224.pth', map_location='cuda')['model']
            state_dict = load_model_and_may_interpolate(state_dict, vision_tower)
            vision_tower.load_state_dict(state_dict, strict=False)
        else:
            vision_tower = self.vision_tower
        self.vision_tower = vision_tower.to(torch.float16)

        train_img_transform, eval_img_transform = build_transform(
            is_train=True, randaug=not no_randaug, input_size=vision_tower.args.img_size), build_transform(is_train=False, input_size=vision_tower.args.img_size)

        if (num_query is not None) and (not hasattr(self, 'query')):
            assert num_query > 2
            bos_weight = vision_tower.beit3.text_embed.weight.data[0]
            eos_weight = vision_tower.beit3.text_embed.weight.data[2]
            query_init_weight = [bos_weight] + [None] * (num_query - 2) + [eos_weight]
            self.query = construct_query_parameter(
                num_query, vision_tower.args.encoder_embed_dim, query_init_weight)

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_tower.hidden_size, self.config.hidden_size)

        return dict(
            image_processor=(train_img_transform, eval_img_transform),
            image_token_len=num_query,
            vision_config=self.vision_config
        )

    def forward(self, images: Optional[torch.FloatTensor] = None):
        vision_tower = getattr(self, 'vision_tower', None)
        attn_mask = None
        if type(images) is list:
            # variable length images
            image_features = []
            for image in images:
                image_forward_out = vision_tower(pixel_values=image.unsqueeze(0), query_embed=self.query, attn_mask=attn_mask)
                image_features.append(image_forward_out)
            image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
        else:
            image_features = vision_tower(pixel_values=images, query_embed=self.query, attn_mask=attn_mask)
            image_features = self.mm_projector(image_features)
        return image_features


class Beit3LlavaVitWrapper(PreTrainedModel):
    config_class = Beit3LlavaConfig

    def __init__(self, config, mm_vision_tower=None):
        super().__init__(config)
        self.model = Beit3Model(config, mm_vision_tower=mm_vision_tower)
        img_processor = build_transform(is_train=False, input_size=int(config.mm_vision_tower.split('_')[-1]))
        self.processor = lambda img_list: torch.stack([img_processor(img) for img in img_list])
        self.post_init()

    def get_vision_embedding(self, images, device='cuda'):
        pixel_values = self.processor(images).to(device)
        ret = self.model(pixel_values)
        return ret

    def get_vision_embedding_per_layer(self, images, device='cuda'):
        pixel_values = self.processor(images).to(device)
        ret = self.model.vision_tower.beit3(visual_tokens=pixel_values, return_all_hiddens=True)
        return ret['encoder_states']

    def get_num_layers(self):
        return len(self.model.vision_tower.beit3.encoder.layers) + 1

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False):
        self.model.vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            self.model.vision_config.im_start_token, self.model.vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        self.model.vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]


class Beit3VitWrapper(nn.Module):
    """This is for the pre-trained Beit-3 model, without aligning with Muffin"""
    def __init__(self, model_path):
        super().__init__()
        self.config = _get_large_config(img_size=224)
        self.beit3 = BEiT3(self.config)
        self.num_img_patches = self.beit3.vision_embed.num_position_embeddings()
        msg = self.load_state_dict(torch.load(model_path)['model'], strict=False)
        print(msg)
        self.beit3.eval()
        img_processor = build_transform(is_train=False, input_size=224)
        self.processor = lambda img_list: torch.stack([img_processor(img) for img in img_list])

    def get_vision_embedding(self, images, device='cuda'):
        pixel_values = self.processor(images).to(device)
        ret = self.beit3(visual_tokens=pixel_values)
        # as beit3 was pre-trained with patch-level masked modelling, the patch tokens contains information from the image.
        return ret['encoder_out'][:, 1:]


class MuffinWrapper(Beit3LlavaLlamaModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        img_processor = build_transform(is_train=False, input_size=int(config.mm_vision_tower.split('_')[-1]))
        self.processor = lambda img_list: torch.stack([img_processor(img) for img in img_list])

    def init_vision_config(self):
        vision_config = self.vision_config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        mm_use_im_start_end = getattr(self.config, "mm_use_im_start_end", False)
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.vision_config = vision_config

        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)


    def get_vision_embedding_per_layer(self, images, device='cuda'):
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        pixel_values = self.processor(images).to(device)
        image_features = self.vision_tower(pixel_values=pixel_values, query_embed=self.query, attn_mask=None)
        image_features = self.mm_projector(image_features)

        # dummy_image_features = torch.zeros(self.config.num_query, self.vision_tower.hidden_size, device=pixel_values.device, dtype=pixel_values.dtype)
        # dummy_image_features = self.mm_projector(dummy_image_features)
        mm_use_im_start_end = getattr(self.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.config.num_query + DEFAULT_IM_END_TOKEN
        else:
            prompt = DEFAULT_IMAGE_PATCH_TOKEN * self.config.num_query

        input_ids = self.tokenizer([prompt] * len(images), return_tensors='pt').input_ids.to(device)
        inputs_embeds = self.embed_tokens(input_ids)

        new_input_embeds = []
        cur_image_idx = 0

        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                continue
            if self.vision_config.use_im_start_end:
                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == self.vision_config.im_start_token).sum() != (cur_input_ids == self.vision_config.im_end_token).sum():
                    raise ValueError("The number of image start tokens and image end tokens should be the same.")
                image_start_tokens = torch.where(cur_input_ids == self.vision_config.im_start_token)[0]
                for image_start_token_pos in image_start_tokens:
                    cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                    num_patches = cur_image_features.shape[0]
                    if cur_input_ids[image_start_token_pos + num_patches + 1] != self.vision_config.im_end_token:
                        raise ValueError("The image end token should follow the image start token.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                    cur_image_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
            else:
                raise NotImplementedError
        assert cur_image_idx == len(image_features), "NOT all input images are used"
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        print("inputs_embeds.shape:", inputs_embeds.size())

        llama_output = super(Beit3LlavaLlamaModel, self).forward(
            input_ids=None, inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)
    
        # start_token_pos is same for every example in the batch
        hidden_states = [t[:, image_start_token_pos + 1: image_start_token_pos + num_patches + 1] for t in llama_output.hidden_states]
        return hidden_states

    def get_vision_embedding(self, images, layer=[-1]):
        image_hidden_states = self.get_vision_embedding_per_layer(images)
        return [image_hidden_states[l] for l in layer]
