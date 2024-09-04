from transformers import PreTrainedModel, PretrainedConfig
from collections import OrderedDict
import math
import requests
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from eval_wrappers.qwen.modeling_qwen import QWenLMHeadModel
from eval_wrappers.qwen.visual import VisionTransformer
from eval_wrappers.qwen.configuration_qwen import QWenConfig


class QwenVisualModel(PreTrainedModel):
    config_class = QWenConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual = VisionTransformer(**config.visual)
        self.post_init()


class QWenViTWrapper(PreTrainedModel):
    config_class = QWenConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = QwenVisualModel(config)
        self.post_init()

    def get_num_layers(self):
        return len(self.transformer.visual.transformer.resblocks) + 1

    def get_vision_embedding(self, images):
        images, hidden_states = self.transformer.visual.encode(images)
        return images

    def get_vision_embedding_per_layer(self, images):
        images, hidden_states = self.transformer.visual.encode(images, output_hidden_states=True)
        hidden_states = [h.permute(1, 0, 2) for h in hidden_states]
        return hidden_states


class QWenVLWrapper(QWenLMHeadModel):
    def get_num_layers(self):
        return len(self.transformer.h) + 1

    def get_vision_embedding_per_layer(self, images):
        assert hasattr(self, 'tokenizer')
        text_list = ['<img>%s</img>' % image for image in images]
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to('cuda')
        outputs = self.transformer(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = [h[:, 1:-1] for h in outputs.hidden_states]
        return hidden_states

    def get_vision_embedding(self, images, layer=[-1]):
        image_hidden_states = self.get_vision_embedding_per_layer(images)
        return [image_hidden_states[l] for l in layer]
