from transformers import Dinov2Model


class DINOV2Wrapper(Dinov2Model):
    def get_num_layers(self):
        return len(self.encoder.layer) + 1

    def get_vision_embedding_per_layer(self, images):
        assert hasattr(self, 'processor')
        model_input = self.processor(images, return_tensors='pt').to(self.device)
        return self(**model_input, output_hidden_states=True, return_dict=True).hidden_states
