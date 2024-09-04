import sys
import json
import random
import numpy as np 


def evaluate(ans_file, label_file):
    answers = [json.loads(q) for q in open(ans_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['answer']

        # if is the result of Muffin, keep the first few chars of the last turn
        if 'Muffin' in ans_file or 'RLHF-V' in ans_file:
            text = '### Assistant: '.join(text.split('### Assistant: ')[1:])
            text = text.split('###')[0]
        elif 'qwen-vl' in ans_file.lower():
            if 'in the image?' in text:
                text = text.split('in the image?')[1]
            elif 'in the imange?' in text:
                text = text.split('in the imange?')[1]
        else:
            # Only keep the first sentence
            if text.find('.') != -1:
                text = text.split('.')[0]

        text = text.strip()
        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    # print('TP\tFP\tTN\tFN\t')
    # print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    # precision = float(TP) / float(TP + FP)
    # recall = float(TP) / float(TP + FN)
    # f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc


if __name__ == '__main__':
    ans_file, label_file = sys.argv[1:]
    if '%s' in ans_file and '%s' in label_file:
        acc_list = list()
        for neg in ['random', 'popular', 'adversarial']:
            acc = evaluate(ans_file % neg, label_file % neg)
            acc_list.append(round(acc, 3))
        print(acc_list)
        print('mean: {}'.format(np.mean(acc_list)))
    else:
        acc = evaluate(ans_file, label_file)
        print('Accuracy: {}'.format(acc))
