import pandas as pd
import numpy as np
from fucset import data_encoding_train, predict_standardization

# input_path 输入csv的路径
# output_path 输出csv的路径
# name_of_target 预测目标名称
# encoder_path 编码文件存储位置
# path_stand_feature 标准化特征的路径
# stand_path 标准化模型的路径
def predict_encoding(input_path, output_path, encoder_path, cols, stand_path, path_stand_feature, name_of_target='Final_Price'):
    # target编码方式
    data = pd.read_csv(input_path)
    data, encoder = data_encoding_train(data, cols['str'], name_of_target)
    encoder.to_csv(encoder_path, encoding='utf-8')

    # 数据标准化
    # tmp = []
    # for type in cols.keys():
    #     for key in cols[type]:
    #         tmp.append(key)
    # tmp.remove(name_of_target)
    data = predict_standardization(data=data, mode='train', target=name_of_target, path_model=stand_path, path_stand_feature=path_stand_feature)

    data.to_csv(output_path, index=False, encoding='utf-8')
    return data