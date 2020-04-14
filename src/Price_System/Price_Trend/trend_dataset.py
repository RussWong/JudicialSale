'''
    File: trend_dataset.py
    Date: 2020/4/4
    Author：herozen
    Version: v1.0


'''

import os
import sys
sys.path.append('./module/')

import pandas as pd
from time_dataset_v1_2 import time_group_dataset

DATA_LEN = 20

def trend_dataset(data_input_path, dataset_path, parameter_path, index_path, data_len=DATA_LEN):
    '''Generate dataset for model train.
    Args:
        data_input_path: string, path of input data.
        dataset_path: string, path of dataset, like './Dataset/'
        parameter_path: string, path of parameter.txt
        index_path: string, path of index.csv
        data_len: int, demand of data length
    Return:
        None

    Note:
        1. 处理重复和缺失，缺失包含数据缺失与时间连续缺失。
        

    '''

    # file read
    data = pd.read_csv(data_input_path)
    with open(parameter_path, 'r') as f:
        i = 0
        for line in f.readlines():
            if i == 0:
                target_name = line.strip('\n')
                i += 1
            elif i == 1:
                time_name = line.strip('\n')
                i += 1
            elif i == 2:
                feature = eval(line.strip('\n'))
                i +=1
            elif i == 3:
                timedelta = eval(line.strip('\n'))
                i +=1
            else:
                group = eval(line.strip('\n'))
                print('Parameter_path Read Finished')
        
    

    dataset = time_group_dataset(data_raw=data, time_name=time_name,
                                 target_name=target_name, group_raw=group,
                                  timedelta=timedelta, data_path_raw=dataset_path,
                                   data_len = data_len)          

    dir_path = os.path.abspath(os.path.join(index_path, "../"))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dataset.to_csv(index_path, index=False)
    print('Dataset Store Finished')


