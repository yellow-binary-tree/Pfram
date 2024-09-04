from transformers import SamModel


class SamWrapper(SamModel):
    def get_num_layers(self):
        return len(self.vision_encoder.layers) + 1

    def get_vision_embedding_per_layer(self, images):
        assert hasattr(self, 'processor')
        model_input = self.processor(images, return_tensors='pt').to(self.device)
        hidden_states = self.vision_encoder(pixel_values=model_input.pixel_values, output_hidden_states=True, return_dict=True).hidden_states
        shape = hidden_states[0].size()
        hidden_states = [h.view(shape[0], shape[1] * shape[2], shape[3]) for h in hidden_states]
        return hidden_states
