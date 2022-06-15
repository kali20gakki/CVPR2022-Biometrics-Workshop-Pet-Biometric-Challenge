from glob import glob
import pandas as pd
import os, cv2
from PIL import Image
import numpy as np
from prettytable import PrettyTable
from copy import deepcopy



data = pd.read_csv('dataset/pet_biometric_challenge_2022/test/test_data.csv', encoding='utf-8')

images_pair = [(imageA_name, imageB_name) for imageA_name, imageB_name in zip(data['imageA'], data['imageB'])]


def get_res(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    res = [pred_score for i, pred_score in enumerate(data['prediction'])]

    return res



def get_sortid(x):
    index = np.argsort(x)
    out = np.zeros_like(index)
    for i, idx in enumerate(index):
        out[idx] = i

    return np.array([out])


def ensemble_by_sort(res_list):
    out_list = []
    ensemble_res = []
    for res in res_list:
        out_list.append(get_sortid(res))

    out_list_ = np.array(deepcopy(out_list))

    all_out = np.concatenate(out_list, 0)

    mean_out = list(np.mean(all_out, 0))
    x_min, x_max = np.min(mean_out), np.max(mean_out)

    for x in mean_out:
        ensemble_res.append((x - x_min) / (x_max - x_min)) 

    out_list_ = np.squeeze(out_list_, 1)
    out_list_ = [list(out) for out in out_list_]
    return ensemble_res, out_list_, list(get_sortid(ensemble_res)[0])


def ensemble(file_path_list, show_len = -1):
    res_list = []
    file_names = []
    for file_path in file_path_list:
        res_list.append(get_res(file_path))
        file_names.append(file_path.split('\\')[-1])
    
    table = PrettyTable()

    merge_res, out_list, merge_index = ensemble_by_sort(res_list)

    res_and_index_list = []
    merge_res_and_index = [[res, i] for res, i in zip(merge_res, merge_index)]

    for res, index in zip(res_list, out_list):
        temp = []
        for r, i in zip(res, index):
            temp.append([r, i])
        res_and_index_list.append(temp)

    for name, res in zip(file_names, res_and_index_list):
        table.add_column(name, res[:show_len])
    table.add_column('ensemble res', merge_res_and_index[:show_len])
    #print(table)
    # with open('merge_table.txt', 'w') as f:
    #     f.write(str(table))

    return merge_res


if __name__ == '__main__':
    # 改为所有结果保存的路径
    all_results_dir = 'test_results' 


    file_path_list = [p for p in glob(all_results_dir + '/*.csv')]
    merge_res = ensemble(file_path_list)
    res = []
    imageA_list = []
    imageB_list = []
    for (imageA, imageB), pred in zip(images_pair, merge_res):
        imageA_list.append(imageA)
        imageB_list.append(imageB)
        res.append(pred)



    d = {'imageA': imageA_list, 'imageB': imageB_list, 'prediction': res}

    df = pd.DataFrame(d)
    
    # 保存 dataframe
    df.to_csv(f'./final_submmit_{len(file_path_list)}res.csv', index=None)
    print(f'final_submmit_{len(file_path_list)}res.csv')

