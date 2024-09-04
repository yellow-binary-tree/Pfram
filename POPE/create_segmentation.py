# create ground truth segmentation for different datasets
import os
import json
import random
import argparse


def create_segmentation(dataset, anno_fname, output_fname, num_imgs=500, min_objs=3):
    obj_anno = json.load(open(anno_fname))
    selected_list = list(obj_anno.items())
    random.shuffle(selected_list)

    res = list()
    for image_id, objects in selected_list:
        if len(objects) >= min_objs:
            image_fname = f'AMBER_{image_id}.jpg' if dataset == 'amber' else f'{image_id}.jpg'
            res.append({'image_id': image_id, 'image': image_fname, 'objects': objects})
        if len(res) == num_imgs: break

    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    with open(output_fname, 'w') as f_out:
        for example in res:
            f_out.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='oi')
    parser.add_argument('--input_fname', type=str, required=True)
    parser.add_argument('--output_fname', type=str, required=True)
    args = parser.parse_args()

    create_segmentation(
        dataset=args.dataset, anno_fname=args.input_fname, output_fname=args.output_fname
    )
