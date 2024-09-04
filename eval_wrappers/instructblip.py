from typing import Optional
import random

import torch
from torch import nn
from transformers import InstructBlipVisionModel, InstructBlipQFormerModel, InstructBlipForConditionalGeneration
from transformers import InstructBlipConfig, PreTrainedModel
from transformers import AutoImageProcessor


class InstructBlipVisualEncoderWrapper(PreTrainedModel):
    config_class = InstructBlipConfig

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        self.vision_model = InstructBlipVisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.post_init()
        self.instructions = [""]

    def get_vision_embedding(self, images):
        assert hasattr(self, "processor"), "You need to set the `processor` function first"
        assert hasattr(self, "qformer_processor"), "You need to set the `qformer_tokenizer` function first"
        pixel_values = self.processor(images).to('cuda')
        qformer_text = [random.choice(self.instructions) for _ in images]
        qformer_input_ids = self.qformer_processor(qformer_text).to(pixel_values.device)

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)

        qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]
        language_model_inputs = self.language_projection(query_output)
        return language_model_inputs

    def get_vision_embedding_per_layer(self, images):
        assert hasattr(self, "processor"), "You need to set the `processor` function first"
        assert hasattr(self, "qformer_processor"), "You need to set the `qformer_tokenizer` function first"
        pixel_values = self.processor(images).to('cuda')
        qformer_text = [random.choice(self.instructions) for _ in images]
        qformer_input_ids = self.qformer_processor(qformer_text).to(pixel_values.device)

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)

        qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True, output_hidden_states=True,
        )
        hidden_states = [h for h in vision_outputs.hidden_states] + [h[:, : query_tokens.size(1), :] for h in query_outputs.hidden_states]
        return hidden_states

    def get_num_layers(self):
        return len(self.vision_model.encoder.layers) + 1 + len(self.qformer.encoder.layer) + 1


class InstructBlipWrapper(InstructBlipForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instructions = [""]

    def get_num_layers(self):
        return len(self.config.text_config.num_layers)

    def get_vision_embedding_per_layer(self, images):
        assert hasattr(self, "processor"), "You need to set the `processor` function first"
        assert hasattr(self, "qformer_processor"), "You need to set the `qformer_tokenizer` function first"
        pixel_values = self.processor(images).to('cuda')
        qformer_text = [random.choice(self.instructions) for _ in images]
        qformer_input_ids = self.qformer_processor(qformer_text).to(pixel_values.device)

        with torch.autocast("cuda"):
            # step 1: forward the images through the vision encoder,
            # to get image embeddings of shape (batch_size, seq_len, hidden_size)
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[0]

            # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
            qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
            query_outputs = self.qformer(
                input_ids=qformer_input_ids,
                attention_mask=qformer_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
            )
            query_output = query_outputs[0][:, : query_tokens.size(1), :]

            # step 3: use the language model, conditioned on the query outputs and the prompt
            language_model_inputs = self.language_projection(query_output)
            language_model_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
            )

            assert hasattr(self, 'tokenizer')
            # here the input id is used to append a bos token only
            input_ids = self.tokenizer([''] * len(images), return_tensors="pt").input_ids.to(language_model_inputs.device)
            input_ids_length = input_ids.size(1) - 1
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_embeds, language_model_inputs], dim=1)
            attention_mask = torch.ones_like(input_ids)
            attention_mask = torch.cat([attention_mask, language_model_attention_mask], dim=1)

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
        image_hidden_states = [h[:, 1+input_ids_length:] for h in output_hidden_states]
        return image_hidden_states

    def get_vision_embedding(self, images, layer=[-1]):
        image_hidden_states = self.get_vision_embedding_per_layer(images)
        return [image_hidden_states[l] for l in layer]
