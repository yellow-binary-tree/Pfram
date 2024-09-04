# 调用LLM回答问题的代码
import os
import io
import sys
import json
import base64
import argparse
from tqdm import tqdm
from PIL import Image
from functools import partial

import torch
from transformers import pipeline
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torchvision.transforms import PILToTensor
from transformers import StoppingCriteria


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_size):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.input_size = input_size

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for o in output_ids:
            o = self.tokenizer.decode(o[self.input_size:], skip_special_tokens=True)
            if all([keyword not in o for keyword in self.keywords]):
                return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question answering with LLMs.")
    parser.add_argument("--task", type=str, default='pope')

    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--model", type=str, default='Salesforce/blip2-flan-t5-xl')
    parser.add_argument("--llava_model_base_name", type=str, help="If not using a llava lora model, set this to None")
    parser.add_argument("--input", type=str, default='output/coco/coco_pope_random.json')
    parser.add_argument("--output", type=str, default='results/coco/blip2-flan-t5-xl/coco_pope_random.json')

    parser.add_argument("--text_first", action='store_true', help='if set, use prompt+image instead of image+prompt as VLM input.')
    parser.add_argument("--system_prompt", default=None, help='If you want to change the system prompt, set this.')
    args = parser.parse_args()
    print(args)

    if args.text_first:
        # only some of the models can easily support text first input.
        assert_flag = False
        if 'internlm-x' in args.model: assert_flag = True
        if 'qwen' in args.model.lower(): assert_flag = True
        if 'llava' in args.model.lower(): assert_flag = True
        assert assert_flag, 'text first input is not supported for the model: {}'.format(args.model)

    if args.system_prompt is not None:
        assert_flag = False
        if 'llava' in args.model.lower(): assert_flag = True
        assert assert_flag, 'changing system prompt is not supported for the model: {}'.format(args.model)


    if 'instructblip' in args.model:
        processor = InstructBlipProcessor.from_pretrained(args.model)
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16)     # 
        model.eval()

    elif 'llava' in args.model:
        sys.path.append('../../utils')
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path 
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates

        model_name = get_model_name_from_path(args.model)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model,
            model_base=args.llava_model_base_name,
            model_name=model_name
        )
        conv_mode = {
            # 'llava-v1.5-7b': 'v1', 'llava-v1.5-13b': 'v1', 'llava-v1.6-vicuna-7b': 'v1', 'llava-v1.6-vicuna-13b': 'v1', 
            'llava-llama-2-7b-chat-lightning-lora-preview': 'llava_llama_2',
        }.get(model_name, 'v1')
        model.eval()
    
    elif 'rlhf-v' in args.model.split('/')[-1].lower() or \
         'muffin-13b' in args.model.split('/')[-1].lower():
        sys.path.append('../../utils')
        from muffin.eval.muffin_vqa import init_muffin, wrap_question_with_default_conv, qa_colloator_fn
        model, image_processor, image_token_len, tokenizer = init_muffin(args.model)
        model.eval()
        collate_fn = partial(qa_colloator_fn, tokenizer=tokenizer, img_transform=image_processor)

    elif 'internlm-xcomposer2' in args.model.split('/')[-1].lower() or 'qwen' in args.model.split('/')[-1].lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
        model = model.eval()

    else:
        image_question_answerer = pipeline(task='visual-question-answering', model=args.model, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)

    if args.task in ['pope', 'object_halbench']:
        input_data = [json.loads(line) for line in open(args.input)]
    elif args.task  == 'mmhal_bench':
        input_data = json.load(open(args.input))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    f_out = open(args.output, 'w')

    line = 0
    for example in tqdm(input_data):
        if 'instructblip' in args.model:
            if args.task == 'pope':
                image = Image.open(os.path.join(args.img_folder, example['image'])).convert('RGB')
                inputs = processor(images=image, text=example['text'], return_tensors="pt").to('cuda')
            elif args.task == 'object_halbench':
                image = Image.open(io.BytesIO(base64.b64decode(example['image']))).convert('RGB')
                inputs = processor(images=image, text=example['question'], return_tensors="pt").to('cuda')
            elif args.task == 'mmhal_bench':
                image = Image.open(os.path.join(args.img_folder, example['image_src'].split('/')[-1])).convert('RGB')
                inputs = processor(images=image, text=example['question'], return_tensors="pt").to('cuda')

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=1024)
            answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        elif 'llava' in args.model:
            img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
            if args.task == 'pope':
                image = Image.open(os.path.join(args.img_folder, example['image'])).convert('RGB')
                if args.text_first:
                    qs = example['text'] + '\n' + img_tokens
                else:
                    qs = img_tokens + '\n' + example['text']
            elif args.task == 'object_halbench':
                image = Image.open(io.BytesIO(base64.b64decode(example['image']))).convert('RGB')
                if args.text_first:
                    qs = example['question'] + '\n' + img_tokens
                else:
                    qs = img_tokens + '\n' + example['question']
            elif args.task == 'mmhal_bench':
                image = Image.open(os.path.join(args.img_folder, example['image_src'].split('/')[-1])).convert('RGB')
                if args.text_first:
                    qs = example['question'] + '\n' + img_tokens
                else:
                    qs = img_tokens + '\n' + example['question']

            image_tensor = process_images([image], image_processor, model.config)[0]
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            if args.system_prompt is not None:
                conv.system = args.system_prompt
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size], max_new_tokens=1024
                )
            answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        elif 'rlhf-v' in args.model.split('/')[-1].lower() or \
             'muffin-13b' in args.model.split('/')[-1].lower():
            if args.task == 'pope':
                image = Image.open(os.path.join(args.img_folder, example['image'])).convert('RGB')
                question = wrap_question_with_default_conv(example['text'], image_token_len)
                data_list = [{'question': question, 'raw_question': example['text'], 'image': image}]
                max_new_tokens = 64
            elif args.task == 'object_halbench':
                image = Image.open(io.BytesIO(base64.b64decode(example['image']))).convert('RGB')
                question = wrap_question_with_default_conv(example['question'], image_token_len)
                data_list = [{'question': question, 'raw_question': example['question'], 'image': image}]
                max_new_tokens = 1024
            elif args.task == 'mmhal_bench':
                image = Image.open(os.path.join(args.img_folder, example['image_src'].split('/')[-1]))
                question = wrap_question_with_default_conv(example['question'], image_token_len)
                data_list = [{'question': question, 'raw_question': example['question'], 'image': image}]
                max_new_tokens = 64

            batch = collate_fn(data_list)
            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, batch['input_ids'].shape[-1])

            with torch.inference_mode():
                output = model.generate(        # add other generation configs as you wish
                    input_ids=batch['input_ids'].cuda(),
                    images=batch['images'].half().cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                )

            question, output_ids = batch['raw_questions'][0], output.sequences[0]
            answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        elif 'internlm-x' in args.model.split('/')[-1].lower():
            if args.task == 'pope':
                query = example['text'] + '<ImageHere>' if args.text_first else '<ImageHere>' + example['text']
                image = os.path.join(args.img_folder, example['image'])
            elif args.task == 'object_halbench':
                query = example['question'] + '<ImageHere>' if args.text_first else '<ImageHere>' + example['question']
                image = Image.open(io.BytesIO(base64.b64decode(example['image']))).convert('RGB')
                image = PILToTensor()(image).unsqueeze(0)
            elif args.task == 'mmhal_bench':
                query = example['question'] + '<ImageHere>' if args.text_first else '<ImageHere>' + example['question']
                image = os.path.join(args.img_folder, example['image_src'].split('/')[-1])

            with torch.cuda.amp.autocast(), torch.inference_mode():
                answer, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)

        elif 'qwen' in args.model.split('/')[-1].lower():
            if args.task == 'pope':
                if args.text_first:
                    query = tokenizer.from_list_format([
                        {'text': example['text']},
                        {'image': os.path.join(args.img_folder, example['image'])},
                    ])
                else:
                    query = tokenizer.from_list_format([
                        {'image': os.path.join(args.img_folder, example['image'])},
                        {'text': example['text']},
                    ])

            inputs = tokenizer(query, return_tensors='pt')
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                pred = model.generate(**inputs)
            answer = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

        else:
            if args.task == 'pope':
                question, image = example['text'], os.path.join(args.img_folder, example['image'])
            elif args.task == 'object_halbench':
                question, image = example['question'], Image.open(io.BytesIO(base64.b64decode(example['image']))).convert('RGB')
            elif args.task == 'mmhal_bench':
                question, image = example['question'], Image.open(os.path.join(args.img_folder, example['image_src'].split('/')[-1])).convert('RGB')

            answer = image_question_answerer({'image': image, 'question': question})[0]['answer']

        if args.task == 'pope':
            res = {'question_id': example['question_id'], 'question': example['text'], 'answer': answer}
        elif args.task == 'object_halbench':
            res = {'image_id': example['image_id'], 'question': example['question'], 'answer': answer}
        elif args.task == 'mmhal_bench':
            res = example
            res['model_answer'] = answer

        f_out.write(json.dumps(res) + '\n')
        line += 1
        if line % 10 == 0:
            f_out.flush()


