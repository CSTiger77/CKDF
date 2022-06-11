import json

import numpy as np


def main(result_json_path):
    with open(result_json_path, 'r') as fr:
        result_data = json.load(fr)
    # print(result_data)
    ncm_avg_result = []
    fc_avg_result = []
    fcls_avg_result = []
    svm_avg_result = []
    for key, value in result_data.items():
        if "task" in key:
            rate = 0
            for predl in value:
                rate += predl[0]
            rate /= len(value)
            if "ncm" in key:
                ncm_avg_result.append(rate)
            elif "fc" in key:
                fc_avg_result.append(rate)
            elif "svm" in key:
                svm_avg_result.append(rate)
            else:
                fcls_avg_result.append(rate)
    return [ncm_avg_result, fc_avg_result, svm_avg_result, fcls_avg_result]


def fc_main(result_json_path):
    with open(result_json_path, 'r') as fr:
        result_data = json.load(fr)
    # print(result_data)
    ncm_avg_result = []
    fc_avg_result = []
    fcls_avg_result = []
    svm_avg_result = []
    for key, value in result_data.items():

        if "task" in key:
            if "fc" not in key:
                temp = []
                for predl in value:
                    temp.append(predl[0])
                fc_avg_result.append(temp)

    return fc_avg_result

if __name__ == "__main__":
    iCaRL_pretrain_extra_result_json_path = r"C:\Users\likunchi\Desktop\pretrain_extra_CIFAR100_1_4_2000.result"
    iCaRL_pretrain_extra_result_json_path_2 = r"C:\Users\likunchi\Desktop\pretrain_extradata_cifar100_1_4_2000.result"
    iCaRL_extra_result_json_path = r"C:\Users\likunchi\Desktop\extra_CIFAR100_1_4_2000.result"
    iCaRL_result_json_path = r"C:\Users\likunchi\Desktop\CIFAR100_1_4_2000.result"
    iCaRL_result_json_path_2 = r"C:\Users\likunchi\Desktop\CIFAR100_1_4_2000_2.result"
    iCaRL_pretrain_result_json_path = r"C:\Users\likunchi\Desktop\pretrain_CIFAR100_1_4_2000.result"
    iCaRL_pretrain_result_json_path_2 = r"C:\Users\likunchi\Desktop\pretrain_CIFAR100_1_4_2000_2.result"
    results = main(iCaRL_result_json_path)
    print(results[1])
    results_2 = main(iCaRL_result_json_path_2)
    print(results_2[1])
    fc_result_per_task = fc_main(iCaRL_result_json_path)
    fc_result_per_task_2 = fc_main(iCaRL_result_json_path_2)
    for i in range(len(fc_result_per_task)):
        print(fc_result_per_task[i], "||", fc_result_per_task_2[i])
        dif = np.array(fc_result_per_task[i]) - np.array(fc_result_per_task_2[i])
        print(dif)
        print(dif.sum())

