import sys
sys.path.append('./utils')

import torch
from transformers import PreTrainedModel, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path 

import torchvision.transforms as T
import random


class LlavaVitWrapper:
    def __init__(self, model, model_base=None):
        model_name = get_model_name_from_path(model)
        _, self.llava, self.image_processor, _ = load_pretrained_model(
            model_path=model,
            model_base=model_base,
            model_name=model_name,
        )
        # self.llava.cuda().eval()
        del self.llava.model.layers
        torch.cuda.empty_cache()

    def get_vision_embedding(self, images):
        images = process_images(images, self.image_processor, self.llava.config).half().cuda()
        # for image in images:
        #     image = (image * 255).clip(0, 255).to(torch.uint8)
        #     image = T.ToPILImage()(image)
        #     image.save(f"./llava-v1-img_input/{random.randint(0, 1000000)}.png")
        return self.llava.encode_images(images)

    def get_vision_embedding_per_layer(self, images):
        images = process_images(images, self.image_processor, self.llava.config).half().cuda()
        vision_model = self.llava.get_model().get_vision_tower().vision_tower
        image_forward_out = vision_model(images, output_hidden_states=True)
        return image_forward_out.hidden_states

    def get_num_layers(self):
        return len(self.llava.get_model().get_vision_tower().vision_tower.vision_model.encoder.layers) + 1

    def to(self, device):
        self.llava.to(device)

class LlavaWrapper:
    """Probes the image representation after LLM"""
    def __init__(self, model, model_base):
        model_name = get_model_name_from_path(model)
        self.tokenizer, self.llava, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model,
            model_base=model_base,
            model_name=model_name
        )
        self.instructions = ['']     # overwrite this if you want to change prompt

    def get_num_layers(self):
        return len(self.llava.model.layers) + 1

    def get_vision_embedding_per_layer(self, images):
        batch_size = len(images)
        images_tensor = process_images(images, self.image_processor, self.llava.config).half().cuda()
        instruction = random.choice(self.instructions)
        input_ids = tokenizer_image_token(prompt=instruction + '<image>', tokenizer=self.tokenizer, return_tensors='pt').cuda()
        instruction_ids_length = len(input_ids) - 1      # the last token is for <image>
        input_ids = torch.stack([input_ids] * batch_size)
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.llava.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            images=images_tensor,
            image_sizes=[image.size for image in images]
        )
        output_hidden_states = self.llava.forward(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            output_hidden_states=True, return_dict=True).hidden_states
        image_hidden_states = torch.stack([h[:, instruction_ids_length:] for h in output_hidden_states])
        return image_hidden_states

    def get_vision_embedding(self, images, layer=[-1]):
        image_hidden_states = self.get_vision_embedding_per_layer(images)
        return [image_hidden_states[l] for l in layer]

    def to(self, device):
        self.llava.to(device)

    def reload_vision_tower(self, model_path):
        from transformers import CLIPVisionModel
        print('loading and overwriting vit weights for pratrained model from %s' % model_path)
        self.llava.model.vision_tower.vision_tower = CLIPVisionModel.from_pretrained(model_path)
        self.llava.model.vision_tower.vision_tower.to('cuda')

    def reload_projector(self, model_path):
        print('loading and overwriting projector weights for pratrained model from %s' % model_path)
        self.llava.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)

    def reload_llama(self, model_path):
        from llava.model.language_model.llava_llama import LlavaLlamaModel
        print('loading and overwriting llama weights for pratrained model from %s' % model_path)
        self.llava.model = LlavaLlamaModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
        self.llava.model.to('cuda')
