
# load data
import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool


def filter_classes(class_names):    # 0622 update: some human-related classes needs to be removed
    unwanted_classes = {'Mammal', 'Animal', 'Land vehicle', 'Vehicle', 'Tire', 'Plant', 'Clothing'}
    class_names = class_names[~class_names['DisplayName'].str.startswith('Human')]      # as human leg, human hair... co-occur too often
    class_names = class_names[~class_names['DisplayName'].isin(unwanted_classes)]
    return class_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)

    parser.add_argument('--min_num_objects', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()
    print(args)

    # load data
    annotations = json.load(open(os.path.join(args.input_folder, 'annotations.json')))
    annotations = [e for e in annotations if e['type'] == 'generative']
    anno_dict = {e['id']: set(e['truth']) for e in annotations if len(e['truth']) >= args.min_num_objects}
    print('num images:', len(anno_dict))

    # check some examples for debugging
    print('print some examples for debugging...')
    random_keys = random.sample(anno_dict.keys(), 10)
    for key in random_keys:
        print(key, anno_dict[key])

    # calculate ground truth similarities
    anno_list = list(anno_dict.values())
    def sim_func(idx):
        def object_similarity(obj_anno1, obj_anno2):
            return len(obj_anno1 & obj_anno2)

        base_obj_anno = anno_list[idx]
        ret = list()
        for i in range(idx+1, len(anno_list)):
            sim = object_similarity(base_obj_anno, anno_list[i])
            ret.append(sim)
        return ret

    with Pool(args.num_workers) as pool:
        sim_results = pool.map_async(sim_func, list(range(len(anno_list)))).get()

    # save the results
    os.makedirs(args.output_folder, exist_ok=True)
    json.dump({key: list(val) for key, val in anno_dict.items()}, open(os.path.join(args.output_folder, 'anno_dict.json'), 'w'), indent=4)
    json.dump(['%s.jpg' % i for i in anno_dict.keys()], open(os.path.join(args.output_folder, 'image_fname.json'), 'w'))

    sim_mat = np.zeros((len(anno_list), len(anno_list)))
    for i, result in enumerate(sim_results):
        sim_mat[i, i] = 100
        for j, sim in enumerate(result):
            sim_mat[i, i+j+1] = sim
            sim_mat[i+j+1, i] = sim
    np.save(os.path.join(args.output_folder, 'sims-object.json'), sim_mat)

    rank = list()
    for sim_vector in range(len(sim_mat)):
        rank.append(np.argsort(sim_mat[sim_vector])[::-1][1:])
    rank = np.stack(rank)
    np.save(os.path.join(args.output_folder, 'rank-object.json'), rank)
