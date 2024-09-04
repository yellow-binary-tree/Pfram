import json
import os
import math
import argparse
from tqdm import trange, tqdm

from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
from peft import PeftModel

from eval_wrappers.qwen_vl import QWenViTWrapper, QWenVLWrapper
from eval_wrappers.blip2 import Blip2ViTWrapper, Blip2Wrapper
from eval_wrappers.instructblip import InstructBlipVisualEncoderWrapper, InstructBlipWrapper
from eval_wrappers.llava import LlavaVitWrapper, LlavaWrapper
from eval_wrappers.internlm_xcomposer2 import InternLMXComposer2VitWrapper, InternLMXComposer2Wrapper
from eval_wrappers.dinov2 import DINOV2Wrapper
from eval_wrappers.clip import CLIPViTWrapper
from eval_wrappers.sam import SamWrapper
from eval_wrappers.mae import MaeWrapper


def calculate(all_img_hidden_states, layer=None, sim_topk=None, is_cls_token=False):
    # hidden_states
    normalized_hidden_states = np.concatenate(all_img_hidden_states, axis=0)
    print("all_img_hidden_states.shape:", normalized_hidden_states.shape)

    # as calculate similarity of all images at once may cause cuda OOM error, we calculate it in chunks
    chunk_size = 500
    for i in trange(0, len(normalized_hidden_states), chunk_size):
        chunk = normalized_hidden_states[i:i+chunk_size]
        img_hidden_states = torch.tensor(chunk).cuda()
        with torch.no_grad():
            norms = torch.norm(img_hidden_states, dim=-1, keepdim=True)
            img_hidden_states = img_hidden_states / norms
            normalized_hidden_states[i:i+chunk_size] = img_hidden_states.cpu().numpy().astype(np.float16)

    print('calculating inner product as cos sim..')
    sims = np.zeros((normalized_hidden_states.shape[0], normalized_hidden_states.shape[0]), dtype=np.float16)
    chunk_size = math.ceil(len(normalized_hidden_states) / args.num_chunks)
    for j in range(0, len(normalized_hidden_states), chunk_size):
        chunk_y = torch.tensor(normalized_hidden_states[j:j+chunk_size]).cuda()      # [chunk_size, num_tokens, hidden_size]
        for i in trange(len(normalized_hidden_states)):
            example = torch.tensor(normalized_hidden_states[i]).cuda()        # [num_tokens, hidden_size]
            chunk_sims = torch.matmul(chunk_y, example.transpose(0, 1).unsqueeze(0))  # [chunk_size, num_tokens, num_tokens]
            chunk_sims = chunk_sims.max(dim=-1).values
            if sim_topk is not None:
                chunk_sims = torch.topk(chunk_sims, k=sim_topk, dim=-1).values
            chunk_sims = chunk_sims.sum(dim=-1)      # [chunk_size]
            sims[j:j+chunk_size, i] = chunk_sims.cpu().numpy().astype(np.float16).T     # this similarity score is not symmetric. max sim of each src patch should be calculated, instead of target patch

    if layer is None:
        if not is_cls_token:
            np.save(os.path.join(args.output_folder, 'sims.npy'), sims)
        else:
            np.save(os.path.join(args.output_folder, 'sims-cls.npy'), sims)
    else:
        if not is_cls_token:
            np.save(os.path.join(args.output_folder, 'sims-layer_%d.npy' % layer), sims)
        else:
            np.save(os.path.join(args.output_folder, 'sims-layer_%d-cls.npy' % layer), sims)

    # rank
    rank = np.argsort(-sims, axis=1)[:, 1:]
    if layer is None:
        if not is_cls_token:
            np.save(os.path.join(args.output_folder, 'rank.npy'), rank)
        else:
            np.save(os.path.join(args.output_folder, 'rank-cls.npy'), rank)
    else:
        if not is_cls_token:
            np.save(os.path.join(args.output_folder, 'rank-layer_%d.npy' % layer), rank)
        else:
            np.save(os.path.join(args.output_folder, 'rank-layer_%d-cls.npy' % layer), rank)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--image_base_folder', type=str, default='data/OpenImages/images')
    parser.add_argument('--image_fnames', type=str, default='outputs/oi/image_fname.json')
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_chunks', type=int, default=8, help="num chunks for calculating similarity. if your GPU memory is small, set this to a larger value")
    parser.add_argument('--add_llm', action='store_true')
    parser.add_argument('--is_vit', action='store_true')
    parser.add_argument('--llm_layer', type=int, nargs='+', default=[])
    parser.add_argument('--get_embed_per_layer', action='store_true')
    parser.add_argument('--vit_final_layer', type=int, default=0, help="if the VLM does not use the last layer of vit as visual input, change this number.")
    parser.add_argument('--sim_topk', type=int, default=None, help="use only the topk simlarity scores when the number of the patches per images is too large")
    parser.add_argument('--instructions', type=str, nargs='+', default=None, help="set this if you want to add text before image for VLM input.")
    parser.add_argument('--output_folder', type=str, required=True)

    parser.add_argument('--llava_model_base_name', type=str, default=None)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.output_folder, exist_ok=True)

    if args.instructions is not None:
        # only some of the wrappers support adding prompt before image now
        assert_flag = False
        if 'llava' in args.model_name.lower(): assert_flag = True
        if 'instructblip' in args.model_name.lower(): assert_flag = True
        assert assert_flag, f"{args.model_name} does not support adding prompt before image"

    elif args.model_name == 'Qwen/Qwen-VL':
        if args.add_llm:
            model = QWenVLWrapper.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            model.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            model.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        else:
            model = QWenViTWrapper.from_pretrained(args.model_name)
            model = model.eval().cuda()

    elif 'blip2' in args.model_name:
        if args.add_llm:
            model = Blip2Wrapper.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
        else:
            model = Blip2ViTWrapper.from_pretrained(args.model_name)
            model.cuda().eval()
        img_processor = AutoImageProcessor.from_pretrained(args.model_name)
        model.processor = lambda x: img_processor(x, return_tensors="pt").pixel_values

    elif 'instructblip' in args.model_name:
        if args.add_llm:
            model = InstructBlipWrapper.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
        else:
            model = InstructBlipVisualEncoderWrapper.from_pretrained(args.model_name)
            model.cuda().eval()
        processor = AutoProcessor.from_pretrained(args.model_name)
        model.processor = lambda x: processor.image_processor(x, return_tensors="pt").pixel_values
        model.qformer_processor = lambda x: processor.qformer_tokenizer(x, truncation=True, padding=True, return_tensors="pt").input_ids
        model.tokenizer = processor.tokenizer
        if args.instructions is not None:
            model.instructions = args.instructions
        print('instructions to choose from:', model.instructions)

    elif 'llava' in args.model_name.lower():
        if args.add_llm:
            model = LlavaWrapper(args.model_name, args.llava_model_base_name)
        else:
            model = LlavaVitWrapper(args.model_name, args.llava_model_base_name)
        model.llava.config.image_aspect_ratio = 'square'        # do not pad the image, use center_crop as default in transformers processors

        if args.add_llm:
            if args.instructions is not None:
                model.instructions = args.instructions
            print('instructions to choose from:', model.instructions)

    elif 'rlhf-v' in args.model_name.split('/')[-1].lower() or \
         'muffin-13b' in args.model_name.split('/')[-1].lower():
        from eval_wrappers.muffin import Beit3LlavaVitWrapper, Beit3VitWrapper, MuffinWrapper, build_transform as muffin_build_transform
        if args.add_llm:
            model = MuffinWrapper.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map='auto')
            model.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model.init_vision_config()
        else:
            model = Beit3LlavaVitWrapper.from_pretrained(args.model_name)
            model.cuda().eval()

    elif 'internlm-xcomposer2' in args.model_name:
        if args.add_llm:
            model = InternLMXComposer2Wrapper.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map='auto')
            model.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        else:
            model = InternLMXComposer2VitWrapper.from_pretrained(args.model_name, is_vit=args.is_vit)
            model.cuda().eval()

    elif 'dinov2' in args.model_name.lower():
        model = DINOV2Wrapper.from_pretrained(args.model_name)
        model.processor = AutoImageProcessor.from_pretrained(args.model_name)
        model.eval().cuda()

    elif 'clip-vit' in args.model_name.lower():
        model = CLIPViTWrapper.from_pretrained(args.model_name)
        model.processor = AutoImageProcessor.from_pretrained(args.model_name)
        model.eval().cuda()

    elif 'sam' in args.model_name.lower():
        model = SamWrapper.from_pretrained(args.model_name)
        model.processor = AutoImageProcessor.from_pretrained(args.model_name)
        model.eval().cuda()

    elif 'mae' in args.model_name.lower():
        model = MaeWrapper.from_pretrained(args.model_name)
        model.processor = AutoImageProcessor.from_pretrained(args.model_name)
        model.eval().cuda()


    if 'llava' in args.model_name.lower():
        model_param_names = list()
        for name, _  in model.llava.named_parameters():
            model_param_names.append(name)
    else:
        model_param_names = list()
        for name, _  in model.named_parameters():
            model_param_names.append(name)

    print("Parameters in this model:", model_param_names)

    image_fnames = [os.path.join(args.image_base_folder, i) for i in json.load(open(args.image_fnames))]
    if args.num_images is not None:
        print('only using the first %d images!' % args.num_images)
        image_fnames = image_fnames[:args.num_images]

    print('%d images to eval' % len(image_fnames))

    if args.get_embed_per_layer:
        all_img_hidden_states = [list() for _ in range(model.get_num_layers())]
    elif len(args.llm_layer):
        all_img_hidden_states = [list() for _ in args.llm_layer]
    else:
        all_img_hidden_states = list()

    for i in trange(0, len(image_fnames), args.batch_size):
        batch_image_fnames = image_fnames[i:i+args.batch_size]
        if 'qwen' in args.model_name.lower():
            batch_images = batch_image_fnames
        else:
            batch_images = [Image.open(i).convert("RGB") for i in batch_image_fnames]

        with torch.no_grad():
            if args.get_embed_per_layer:    # get the visual embedding for every layer in the visual encoder
                outputs = model.get_vision_embedding_per_layer(batch_images)      # [batch_size, num_tokens, hidden_size]
                selected_layers = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
                selected_layers = set([i - args.vit_final_layer for i in selected_layers])

                for layer_idx, layer_outputs in enumerate(outputs):
                    if selected_layers is None or layer_idx in selected_layers:
                        all_img_hidden_states[layer_idx].append(layer_outputs.to(torch.float16).cpu().numpy().astype(np.float16))

            else:
                if args.add_llm:
                    outputs = model.get_vision_embedding(batch_images, layer=args.llm_layer)      # [batch_size, num_tokens, hidden_size]
                    for i in range(len(outputs)):
                        all_img_hidden_states[i].append(outputs[i].to(torch.float16).cpu().numpy().astype(np.float16))
                else:       # DEPRECATED: only get the image rep of LLM layer 0. You can also use `--add_llm --llm_layer 0 `to get the same results
                    outputs = model.get_vision_embedding(batch_images)      # [batch_size, num_tokens, hidden_size]
                    all_img_hidden_states.append(outputs.to(torch.float16).cpu().numpy().astype(np.float16))

    if 'llava' in args.model_name.lower():
        del model.llava
    del model
    torch.cuda.empty_cache()

    # all_img_hidden_states = np.load(os.path.join(args.output_folder, 'hidden_states.npy'))
    if args.get_embed_per_layer:
        for i, layer_hidden_states in enumerate(all_img_hidden_states):
            if len(layer_hidden_states):
                if args.is_vit and not 'qwen' in args.model_name.lower():
                    # if the tested model is a ViT that has CLS token, calculate CLS token and patch tokens seperately
                    calculate([h[:, 0:1] for h in layer_hidden_states], sim_topk=args.sim_topk, layer=i, is_cls_token=True)
                    calculate([h[:, 1:] for h in layer_hidden_states], sim_topk=args.sim_topk, layer=i, is_cls_token=False)
                else:               # else, calculate all tokens at the same time
                    calculate(layer_hidden_states, sim_topk=args.sim_topk, layer=i)
    elif len(args.llm_layer):
        for layer, layer_hidden_state in zip(args.llm_layer, all_img_hidden_states):
            calculate(layer_hidden_state, sim_topk=args.sim_topk, layer=layer)
    else:
        calculate(all_img_hidden_states, sim_topk=args.sim_topk)
