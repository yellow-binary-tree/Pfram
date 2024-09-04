import torch
from typing import Optional
from transformers import ViTMAEForPreTraining


class MaeWrapper(ViTMAEForPreTraining):
    def __init__(self, config):
        print('setting mask ratio to 0 for inference')
        config.mask_ratio = 0
        super().__init__(config)


    def get_num_layers(self):
        return len(self.vit.encoder.layer) + 1 + len(self.decoder.decoder_layers)

    def _get_vision_embedding_per_layer(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ):
        all_hidden_states = list()
        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden_states.extend(outputs.hidden_states)
        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask
        decoder_outputs = self.decoder(
            latent, ids_restore,
            output_hidden_states=True, return_dict=True
        )
        all_hidden_states.extend(decoder_outputs.hidden_states[1:])     # discard the input layer of decoder, as its almost the same as the output layer of the encoder
        return all_hidden_states

    def get_vision_embedding_per_layer(self, images):
        assert hasattr(self, 'processor')
        model_input = self.processor(images, return_tensors='pt').to(self.device)
        hidden_states = self._get_vision_embedding_per_layer(pixel_values=model_input.pixel_values)
        return hidden_states
