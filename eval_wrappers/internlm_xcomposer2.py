import re

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import PretrainedConfig, PreTrainedModel
from transformers import CLIPVisionModel
from eval_wrappers.internlm.modeling_internlm2 import InternLM2Model, InternLM2PreTrainedModel
from eval_wrappers.internlm.build_mlp import build_vision_projector, build_vision_tower
from eval_wrappers.internlm.configuration_internlm_xcomposer2 import InternLMXcomposer2Config


class InternLMXComposer2VitWrapper(PreTrainedModel):
    config_class = InternLMXcomposer2Config

    def __init__(self, config, is_vit=False):
        super().__init__(config)
        self.post_init()

        print('is_vit =', is_vit)
        self.vit = build_vision_tower(select_feature='cls_patch' if is_vit else 'patch')
        self.vision_proj = build_vision_projector()

        self.vis_processor = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_num_layers(self):
        return len(self.vit.vision_tower.vision_model.encoder.layers) + 1

    def get_vision_embedding(self, images):
        images = torch.stack([self.vis_processor(image).to('cuda') for image in images])
        img_embeds = self.vit(images)
        img_embeds = self.vision_proj(img_embeds)
        return img_embeds

    def get_vision_embedding_per_layer(self, images):
        images = torch.stack([self.vis_processor(image).to('cuda') for image in images])
        outputs = self.vit.vision_tower(images, output_hidden_states=True)
        output_hidden_states = outputs.hidden_states
        return output_hidden_states

    # def cuda(self):
    #     self.vit.cuda()
    #     return self
    
    # def eval(self):
    #     self.vit.eval()
    #     return self

class InternLMXComposer2Wrapper(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'
    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = None

        self.max_length = config.max_length
        print(f'Set max length to {self.max_length}')
        # Initialize weights and apply final processing
        self.post_init()

        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()

        self.vis_processor = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def interleav_wrap(self, img_list, text_list):
        wrap_embeds_list, wrap_atts_list = [], []
        wrap_target_list, wrap_im_mask_list = [], []

        for image, text in zip(img_list, text_list):
            img_embeds, atts_img, img_target = self.img2emb(image.unsqueeze(0))
            text = text[0]
            parts = text.split('<ImageHere>')
            wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
            temp_len = 0
            image_nums, im_len = img_embeds.shape[:2]
            need_bos = True
            for idx, part in enumerate(parts):
                if len(part) > 0:
                    part_tokens = self.tokenizer(
                        part,
                        return_tensors='pt',
                        padding='longest',
                        add_special_tokens=need_bos).to('cuda:0')  # .to(self.device)
                    if need_bos:
                        need_bos = False
                    wrap_tokens.append(part_tokens.input_ids)
                    part_embeds = self.model.tok_embeddings(
                        part_tokens.input_ids)
                    wrap_embeds.append(part_embeds)
                    wrap_atts.append(part_tokens.attention_mask)
                    wrap_im_mask.append(
                        torch.zeros(part_embeds.shape[:2]).to('cuda:0'))  # .to(self.device)

                    temp_len += part_embeds.shape[1]
                if idx < image_nums:
                    wrap_tokens.append(img_target[idx].unsqueeze(0))
                    wrap_embeds.append(img_embeds[idx].unsqueeze(0))
                    wrap_atts.append(atts_img[idx].unsqueeze(0))
                    wrap_im_mask.append(
                        torch.ones_like(atts_img[idx].unsqueeze(0)))

                    temp_len += im_len
                if temp_len > self.max_length:
                    break

            wrap_tokens, wrap_embeds = [w.to('cuda:0') for w in wrap_tokens], [w.to('cuda:0') for w in wrap_embeds]
            wrap_atts, wrap_im_mask = [w.to('cuda:0') for w in wrap_atts], [w.to('cuda:0') for w in wrap_im_mask]
            wrap_tokens = torch.cat(wrap_tokens, dim=1)
            wrap_embeds = torch.cat(wrap_embeds, dim=1)
            wrap_atts = torch.cat(wrap_atts, dim=1)
            wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

            # wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)
            wrap_target = wrap_tokens       # no human targets here

            wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
            wrap_atts = wrap_atts[:, :self.max_length].to(self.device)
            wrap_target = wrap_target[:, :self.max_length].to(self.device)
            wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device)

            wrap_embeds_list.append(wrap_embeds)
            wrap_atts_list.append(wrap_atts)
            wrap_target_list.append(wrap_target)
            wrap_im_mask_list.append(wrap_im_mask)

        wrap_embeds = torch.cat(wrap_embeds_list)
        wrap_atts = torch.cat(wrap_atts_list)
        wrap_target = torch.cat(wrap_target_list)
        wrap_im_mask = torch.cat(wrap_im_mask_list)
        return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask

    def img2emb(self, image):
        img_embeds = self.vision_proj(self.vit(image.to(self.device)))
        atts_img = torch.ones(
            img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        img_target = torch.ones(
            img_embeds.size()[:2], dtype=torch.long).to(
                img_embeds.device) * -100

        return img_embeds, atts_img, img_target

    def get_num_layers(self):
        return len(self.model.layers) + 1

    def get_vision_embedding_per_layer(self, images):
        images = torch.stack([self.vis_processor(image).to('cuda').to(torch.float16) for image in images])
        text_list = ['<ImageHere>'] * len(images)
        to_regress_embeds, attention_mask, targets, im_mask = self.interleav_wrap(images, text_list)

        inputs_embeds = to_regress_embeds[:, :self.max_length]
        attention_mask = attention_mask[:, :self.max_length]
        targets = targets[:, :self.max_length]
        im_mask = im_mask[:, :self.max_length].bool()
        labels = targets

        outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, im_mask=im_mask,
            output_hidden_states=True, return_dict=True,
        )
        hidden_states = [h[:, 1:] for h in outputs.hidden_states]
        return hidden_states

    def get_vision_embedding(self, images, layer=[-1]):
        image_hidden_states = self.get_vision_embedding_per_layer(images)
        return [image_hidden_states[l] for l in layer]
