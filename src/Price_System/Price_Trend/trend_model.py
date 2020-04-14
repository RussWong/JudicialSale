'''
    File: trend_model.py
    Date: 2020/4/
    Author：herozen
    Version: v1.0


'''

import pandas as pd

# output_models_path = '../../../output/Price_System/Price_Trend/models/'
# output_files_path = '../../../output/Price_System/Price_Trend/results/'


def trend_model(dataset_path, index_path, models_path, files_path):
    '''Train model and predict
    Args:
        dataset_path:string, path of dataset, like './dataset/'
        index_path: string, path of index.csv
        models_path: string, path of models, like './models/'
        files_path: string, path of files, like './results/
    Return:
        None

    Note:
        1. 
        

    '''

    # 观察数据集列表
    # index
    index = pd.read_csv(index_path)
    index


    # 数据集读取
    num = 1    # 通过index选出想读取的数据集
    data_path = dataset_path + 'dataset_' + str(num) + '.csv'
    data = pd.read_csv(dataset_path+'dataset_1.csv', index_col=0)
    # data.head()



    # 训练模型


    print('Model Train Finished')

    # 存储趋势图


    print('Files Store Finished')