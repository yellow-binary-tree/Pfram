import os
import json
import numpy as np
import argparse


VIT_LAYERS = {
    'instructblip-vicuna-7b': 40, 'instructblip-vicuna-13b': 40
}

METRIC = 'knn'
NUM_EXAMPLES = 1600
RESULTS_OUTPUT_FILE = 'stat/stat_result.json'

GOLD_FILE = 'output/oi/rank-object.npy'
OUTPUT_BASE_FOLDER = 'output/oi'
OUTPUT_MODEL_FOLDERS = {
    'instructblip-vicuna-7b': {
        'visual_encoder': ['instructblip-vicuna-7b-visual_encoder', 'instructblip-vicuna-7b-qformer'],
        'llm': ['instructblip-vicuna-7b-llm'],
    },
    'instructblip-vicuna-13b': {
        'visual_encoder': ['instructblip-vicuna-13b-visual_encoder', 'instructblip-vicuna-13b-qformer'],
        'llm': ['instructblip-vicuna-13b-llm'],
    },
    'llava-v1.5-7b': {
        'visual_encoder': ['llava-v1.5-7b-visual_encoder'],
        'llm': ['llava-v1.5-7b-llm'],
    },
    'llava-v1.5-13b': {
        'visual_encoder': ['llava-v1.5-7b-visual_encoder'],
        'llm': ['llava-v1.5-13b-llm'],
    },
    'internlm-xcomposer2-vl-7b': {
        'visual_encoder': ['internlm-xcomposer2-vl-7b-visual_encoder'],
        'llm': ['internlm-xcomposer2-vl-7b-llm'],
    },
    'Muffin-13B': {
        'visual_encoder': ['Muffin-13B-visual_encoder'],
        'llm': ['Muffin-13B-llm'],
    },
    'RLHF-V': {
        'visual_encoder': ['RLHF-V-visual_encoder'],
        'llm': ['RLHF-V-llm'],
    },
    'llava-v1.6-vicuna-7b': {
        'visual_encoder': ['llava-v1.6-vicuna-7b-visual_encoder'],
        'llm': ['llava-v1.6-vicuna-7b-llm'],
    },
    'llava-v1.6-vicuna-13b': {
        'visual_encoder': ['llava-v1.6-vicuna-13b-visual_encoder'],
        'llm': ['llava-v1.6-vicuna-13b-llm'],
    },
    'Qwen-VL': {
        'visual_encoder': ['Qwen-VL-visual_encoder'],
        'llm': ['Qwen-VL-llm'],
    },

    # visual encoders
    # 'dinov2-small': {'visual_encoder': ['dinov2-small'], 'llm': []},
    # 'dinov2-base': {'visual_encoder': ['dinov2-base'], 'llm': []},
    # 'dinov2-giant': {'visual_encoder': ['dinov2-giant'], 'llm': []},
    # 'clip-vit-base-patch16': {'visual_encoder': ['clip-vit-base-patch16'], 'llm': []},
    # 'clip-vit-base-patch32': {'visual_encoder': ['clip-vit-base-patch32'], 'llm': []},
    # 'clip-vit-large-patch14': {'visual_encoder': ['clip-vit-large-patch14'], 'llm': []},
    # 'clip-vit-large-patch14-336': {'visual_encoder': ['clip-vit-large-patch14-336'], 'llm': []},

    # 'vit-mae-base': {'visual_encoder': ['vit-mae-base'], 'llm': []},
    # 'vit-mae-large': {'visual_encoder': ['vit-mae-large'], 'llm': []},
    # 'vit-mae-huge': {'visual_encoder': ['vit-mae-huge'], 'llm': []},
    # 'sam-vit-base': {'visual_encoder': ['sam-vit-base'], 'llm': []},
    # 'sam-vit-large': {'visual_encoder': ['sam-vit-large'], 'llm': []},
    # 'sam-vit-huge': {'visual_encoder': ['sam-vit-huge'], 'llm': []},
}


def select_first_images(anno_rank_all, num_images):
    if num_images is None:
        return anno_rank_all
    new_anno_rank_all = list()
    for line in anno_rank_all[:num_images]:
        new_anno_rank_all.append([i for i in line if i < num_images])
    return np.array(new_anno_rank_all)


def stat_results_iou(anno_rank_all, model_rank_all):
    res_dict = dict()
    k = 10
    num_images = len(anno_rank_all)
    while True:
        k = min(num_images, k)
        anno_rank, model_rank = anno_rank_all[:, :k], model_rank_all[:, :k]
        intersection_num_list = list()
        for x, y in zip(anno_rank, model_rank):
            intersection_num_list.append(len(set(x) & set(y)) / k)
        # print(f" {k} - )
        res = float(f"{np.mean(intersection_num_list)*100:.2f}")
        res_dict[k] = res
        if k == num_images: break
        k = 25 if k == 10 else k * 2
    return res_dict


def dcg(scores, k):
    """
    Compute DCG for a given ranking.

    :param scores: List of relevance scores in the order of the ranked items.
    :param k: Number of top-ranked items to consider.
    :return: DCG value.
    """
    scores = scores[:k]
    return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))


def ndcg(scores, k):
    """
    Compute NDCG for a given ranking.

    :param scores: List of relevance scores in the order of the ranked items.
    :param ideal_scores: List of relevance scores in the ideal ranking order.
    :param k: Number of top-ranked items to consider.
    :return: NDCG value.
    """
    ideal_scores = np.sort(scores)[::-1]
    dcg_value = dcg(scores, k)
    idcg_value = dcg(ideal_scores, k)
    if idcg_value == 0:
        return 0.0
    return dcg_value / idcg_value


def stat_results_ndcg(anno_iou_all, model_rank_all):
    res_dict = dict()
    num_images = len(anno_iou_all)
    k = 10
    while True:
        k = min(num_images, k)
        ndcg_values_list = list()
        for i in range(num_images):
            anno_iou = anno_iou_all[i]
            model_rank = model_rank_all[i]
            anno_iou = anno_iou[model_rank]
            ndcg_value = ndcg(anno_iou, k)
            ndcg_values_list.append(ndcg_value)
        res_dict[k] = float(f"{np.mean(ndcg_values_list)*100:.2f}")
        if k == num_images: break
        k = 25 if k == 10 else k * 2
    return res_dict


def stat_results(anno_iou_all, model_rank_all):
    if METRIC == 'knn':
        return stat_results_iou(anno_iou_all, model_rank_all)
    elif METRIC == 'ndcg':
        return stat_results_ndcg(anno_iou_all, model_rank_all)
    else:
        raise ValueError(f'unknown metric {METRIC}, must be one of: ["knn", "ndcg"]')


if __name__ == '__main__':
    print('checking:', GOLD_FILE, OUTPUT_BASE_FOLDER)

    anno_rank_all = np.load(GOLD_FILE)
    anno_rank_all = select_first_images(anno_rank_all, NUM_EXAMPLES)
    results = dict()

    f_out = open(RESULTS_OUTPUT_FILE, 'a')
    for model_name, model_folders in OUTPUT_MODEL_FOLDERS.items():
        results[model_name] = {'visual_encoder': dict(), 'llm': dict()}
        for folder_i, folder in enumerate(model_folders['visual_encoder']):
            if not os.path.exists(os.path.join(OUTPUT_BASE_FOLDER, folder)):
                continue
            files = os.listdir(os.path.join(OUTPUT_BASE_FOLDER, folder))
            no_exist = 0
            for layer in range(100000):
                if f'rank-layer_{layer}.npy' in files:
                    fname = os.path.join(OUTPUT_BASE_FOLDER, folder, f'rank-layer_{layer}.npy')
                    print(fname)
                    model_rank_all = np.load(fname)
                    model_rank_all = select_first_images(model_rank_all, NUM_EXAMPLES)
                    real_layer = layer if folder_i == 0 else layer + VIT_LAYERS[model_name]     # for the case that ViT and Qformer are seperately calculated
                    results[model_name]['visual_encoder'][real_layer] = stat_results(anno_rank_all, model_rank_all)
                    no_exist = 0
                else:
                    no_exist += 1
                    if no_exist > 10:
                        break

        base_folder = model_folders['llm'][0]
        if not os.path.exists(os.path.join(OUTPUT_BASE_FOLDER, base_folder)):
            pass
        else:
            no_exist = 0
            for layer in range(100000):
                if no_exist > 10:
                    break
                fnames = os.listdir(os.path.join(OUTPUT_BASE_FOLDER, base_folder))
                if f'layer_{layer}' in fnames:
                    fname = os.path.join(OUTPUT_BASE_FOLDER, base_folder, f'layer_{layer}', 'rank.npy')
                    print(fname)
                    model_rank_all = np.load(fname)
                    model_rank_all = select_first_images(model_rank_all, NUM_EXAMPLES)
                    results[model_name]['llm'][layer] = stat_results(anno_rank_all, model_rank_all)
                    no_exist = 0
                elif f'rank-layer_{layer}.npy' in fnames:
                    fname = os.path.join(OUTPUT_BASE_FOLDER, base_folder, f'rank-layer_{layer}.npy')
                    print(fname)
                    model_rank_all = np.load(fname)
                    model_rank_all = select_first_images(model_rank_all, NUM_EXAMPLES)
                    results[model_name]['llm'][layer] = stat_results(anno_rank_all, model_rank_all)
                    no_exist = 0
                else:
                    no_exist += 1

        f_out.write(model_name + ' ' + GOLD_FILE + '\n' + json.dumps(results[model_name]) + '\n')
        f_out.flush()

    f_out.write(GOLD_FILE + '\n' + json.dumps(results) + '\n')
    f_out.close()