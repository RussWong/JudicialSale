'''
    File: trend_qualified.py
    Date: 2020/4/3
    Author：herozen
    Version: v1.0


'''

import os
import sys
import pandas as pd

# data_path = '../../../Data/Price_System/Price_Predict/raw/house/used_house_data.csv'

def trend_qualified(data_input_path, data_output_path, parameter_path):
    '''Qualify Data, including duplicates, missing.
    Args:
        data_input_path: string, path of input data.
        data_output_path: string, path of output data.
        parameter_path: string, path of parameter.txt
    Return:
        None

    Note:
        1. 处理重复和缺失，缺失包含数据缺失与时间连续缺失。
        

    '''

    data = pd.read_csv(data_input_path)
    with open(parameter_path, 'r') as f:
        i = 0
        for line in f.readlines():
            if i != 2:
                i += 1
            elif i ==2:
                feature = eval(line.strip('\n'))
                break
    # 去重
    print('==== 数据去重开始 ====') 
    print('去重前数据规格：'+str(data.shape))     
    data.drop_duplicates(keep='first', inplace=True)    
    data.reset_index(drop=True,inplace=True)    
    print('去重后数据规格：'+str(data.shape))    
    print('==== 数据去重结束 ====')

    # 特征调整
    data = data[feature].copy()
    print('特征调整后数据规格：'+str(data.shape)) 

    # 缺失处理
    # 数据缺失
    print('==== 数据缺失处理开始 ====') 
    data.dropna(inplace=True)       
    data.reset_index(drop=True, inplace=True)
    print('处理后数据规格：'+str(data.shape))   
    print('==== 数据缺失处理结束 ====') 
    # 连续缺失
    ## 需设计处理方案，暂时略

    dir_path = os.path.abspath(os.path.join(data_output_path, "../"))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data.to_csv(data_output_path, index=False)



