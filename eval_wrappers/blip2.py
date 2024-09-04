from typing import Optional

import torch
from torch import nn
from transformers import Blip2VisionModel, Blip2QFormerModel, Blip2ForConditionalGeneration
from transformers import Blip2Config, PreTrainedModel
from transformers import AutoImageProcessor

class Blip2ViTWrapper(PreTrainedModel):
    config_class = Blip2Config

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.post_init()

    def get_vision_embedding(self, images):
        assert hasattr(self, "processor"), "You need to set the `processor` function first"
        pixel_values = self.processor(images).to('cuda')

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        return language_model_inputs

    def get_vision_embedding_per_layer(self, images):
        # return the embeddings of every Q-Former layer
        assert hasattr(self, "processor"), "You need to set the `processor` function first"
        pixel_values = self.processor(images).to('cuda')

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True, output_hidden_states=True,
        )
        return query_outputs.hidden_states      # this already contains last hidden state

    def get_num_layers(self):
        return len(self.qformer.encoder.layer) + 1


class Blip2Wrapper(Blip2ForConditionalGeneration):
    def get_num_layers(self):
        return len(self.config.text_config.num_layers)

    def get_vision_embedding_per_layer(self, images):
        assert hasattr(self, "processor"), "You need to set the `processor` function first"
        pixel_values = self.processor(images).to("cuda", torch.float16)

        with torch.autocast("cuda"):
            # step 1: forward the images through the vision encoder,
            # to get image embeddings of shape (batch_size, seq_len, hidden_size)
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[0]

            # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
            )
            query_output = query_outputs[0]

            # step 3: use the language model, conditioned on the query outputs and the prompt
            language_model_inputs = self.language_projection(query_output)
            language_model_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
            )
            inputs_embeds = language_model_inputs
            attention_mask = language_model_attention_mask

            if self.config.use_decoder_only_language_model:
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True,
                )
            else:
                outputs = self.language_model.encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True,
                )
        output_hidden_states = outputs.hidden_states
        image_hidden_states = torch.stack([h[:, 1:] for h in output_hidden_states])
        return image_hidden_states

    def get_vision_embedding(self, images, layer=None):
        if layer is None:
            layer = -1
        image_hidden_states = self.get_vision_embedding_per_layer(images)
        return image_hidden_states[layer]
