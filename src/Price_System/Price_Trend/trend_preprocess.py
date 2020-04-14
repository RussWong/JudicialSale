'''
    File: trend_preprocess.py
    Date: 2020/4/4
    Author： herozen
    Version: v1.0


'''

import os
import sys
src_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
general_path = os.path.abspath(os.path.join(src_path, "General/"))
sys.path.append(general_path)
sys.path.append('./module/')


import pandas as pd
from data_overiew import plot_dist, get_unicount
from missing_analysis import missing_analysis
from time_dataset_v1_2 import time_group_scope


# data_path = '../../../Data/Price_System/Price_Predict/raw/house/used_house_data.csv'

def trend_preprocess(data_path, parameter_path):
    '''Preprocess
    Args:
        data_path: path of data file (csv)
        parameter_path: path of parameters' file
    Return:
        None

    Note:
        1. 确定time_name和target_name
        2. 观察粒度划分
        3. 观察连续性时间分组
        4. 思考连续时间缺失处理方案
        

    '''

    data = pd.read_csv(data_path)
    is_need = False

    # check file
    if not os.path.exists(parameter_path):
        is_need = True

    if is_need:
        print(data.shape)
        data.head()

        # 确定time_name和target_name
        time_name = 'Transaction_Time'
        target_name = 'Final_Price'

        # 数据观察
        # 独立值
        get_unicount(data)  
        # 分布情况 
        plot_dist(data)      
        
        # 缺失情况
        data.isnull().sum()
        data[[time_name,target_name]].isnull().sum()
        # missing_analysis.col_isnull(data, threshould=0.2, isNeed_plot=True)
        # missing_analysis.col_isnull(data[[time_name,target_name]], threshould=0, isNeed_plot=True)

        # 结合感兴趣的特征再进一步数据观察
        feature = ['Region', 'Road','Community_Name', target_name, time_name]
        get_unicount(data[feature]) 
        plot_dist(data[feature]) 
        data[feature].isnull().sum()

        # 留下特征
        data = data[feature].copy()
        # data.dropna(inplace=True)       # 
        # data.reset_index(drop=True, inplace=True)
        print(data.shape)
        data.head()

        # 粒度观察
        group = {}
        group['Region'] = ['徐汇', '长宁']
        group['Road'] = ['南桥', '南京西路']
        # 时间间隔划分
        timedelta = [7, 15]
        time_index = time_group_scope(data_raw=data, group_raw=group, time_name=time_name, timedelta=timedelta)
        time_index

        # 注意此时仍存在数据缺失
        data.isnull().sum()

        # 存储参数
        dir_path = os.path.abspath(os.path.join(parameter_path, "../"))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(parameter_path, 'w') as f:
            f.write(target_name + '\n')
            f.write(time_name + '\n')
        #     feature_str = str(feature).replace('[','').replace(']','').replace("'",'').replace(',','')
            f.write(str(feature) + '\n')
            f.write(str(timedelta) + '\n')
            f.write(str(group) + '\n')
            print('Parameter_path Store Finished')
    else:
        print('Parameter_path Already Store')



