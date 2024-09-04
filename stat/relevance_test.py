import sys
sys.path.append('.')
import numpy as np
from scipy.stats import pearsonr
from functools import partial


from stat.data import (
    OI_OBJECT, OI_POPE_ACC, OI2_OBJECT, OI2_POPE_ACC, OI3_OBJECT, OI3_POPE_ACC,
)


def llm_input(k, llm_result):
    return llm_result['llm']['0'][k]


def llm_max(k, llm_result):
    return max([i[k] for i in llm_result['llm'].values()])


def llm_mean(k, llm_result):
    return np.mean([i[k] for i in llm_result['llm'].values()])

# NOTE: change this function to use different layers and k
llm_value_func = partial(llm_input, '100')

def relevance_test():
    x_data, y_data, keys = list(), list(), list()

    for key in OI_OBJECT:
        if key not in OI_POPE_ACC: continue
        x_data.append(llm_value_func(OI_OBJECT[key]) + llm_value_func(OI2_OBJECT[key]) + llm_value_func(OI3_OBJECT[key]))
        y_data.append(np.mean(OI_POPE_ACC[key]) + np.mean(OI2_POPE_ACC[key]) + np.mean(OI3_POPE_ACC[key]))

        keys.append(key)

    corr_coef, p_value = pearsonr(x_data, y_data)
    print('n: %d, corr_coef: %.4f, p_value: %.4f' % (len(x_data), corr_coef, p_value))


def stat_pfram_results_latex_format():
    # this function is used for generating latex table
    res_dict = dict()
    k_list = [25, 100]
    k_list = [200, 400]
    llm_value_funcs = [partial(llm_input, str(k_list[0])), partial(llm_mean, str(k_list[0])), partial(llm_max, str(k_list[0])), partial(llm_input, str(k_list[1])), partial(llm_mean, str(k_list[1])), partial(llm_max, str(k_list[1]))]


    PFRAM_SCORES = [OI_OBJECT, OI2_OBJECT, OI3_OBJECT]
    POPE_SCORES = [OI_POPE_ACC, OI2_POPE_ACC, OI3_POPE_ACC]

    output_std = len(PFRAM_SCORES) > 1

    print(f'& acc & k={k_list[0]} input & k={k_list[0]} mean & k={k_list[0]} max & k={k_list[1]} input & k={k_list[1]} mean & k={k_list[1]} max \\\\')
    print('\\hline')
    for key in PFRAM_SCORES[0]:
        flag = True
        for score in PFRAM_SCORES + POPE_SCORES:
            if key not in score:
                flag = False
                break
        if not flag: continue
        res_dict[key] = list()

        pope_acc = [np.mean(score[key]) * 100 for score in POPE_SCORES]
        if output_std:
            res_dict[key].append('%.2f (%.2f)' % (round(np.mean(pope_acc), 2), round(np.std(pope_acc), 2)))
        else:
            res_dict[key].append('%.2f' % round(np.mean(pope_acc), 2))

        for i, llm_value_func in enumerate(llm_value_funcs):
            pfram_scores = [llm_value_func(score[key]) for score in PFRAM_SCORES]
            if output_std:
                res_dict[key].append('%.2f (%.2f)' % (round(np.mean(pfram_scores), 2), round(np.std(pfram_scores), 2)))
            else:
                res_dict[key].append('%.2f' % round(np.mean(pfram_scores), 2))

    for key in res_dict:
        print(key, '&', ' & '.join(res_dict[key]) + ' \\\\')
    print('\\hline')

    # stat the pearsons's r and p value
    key = "Pearson's r (p value)"
    pearsonr_list = []
    pearsonr_list.append('%.2f (%.2f)' % (100, 0))
    for i in range(len(llm_value_funcs)):
        x_data = [float(l[0].split('(')[0]) for l in res_dict.values()]
        y_data = [float(l[i+1].split('(')[0]) for l in res_dict.values()]
        corr_coef, p_value = pearsonr(x_data, y_data)
        if p_value > 0.05:
            pearsonr_list.append('%.2f (%.2f)' % (corr_coef*100, p_value))
        else:
            pearsonr_list.append('%.2f (%.2f)*' % (corr_coef*100, p_value))
    print(key, '&', ' & '.join(pearsonr_list) + ' \\\\')


if __name__ == '__main__':
    relevance_test()