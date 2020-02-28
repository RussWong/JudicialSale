import pandas as pd
import numpy as np
from fucset import data_encoding, data_encoding_2

# input_path 输入csv的路径
# output_path 输出csv的路径
# cols 输入csv的标签分类信息{'num':[],'str':[]}
# type_of_encoding  # 目前支持one-hot target两种编码方式
# name_of_target 预测目标名称
def predict_encoding(input_path, output_path, cols, name_of_target='Final_Price', type_of_encoding='target'):

    data = pd.read_csv(input_path)
    if type_of_encoding == 'target':
        data = data_encoding_2(data, cols, name_of_target)
    elif type_of_encoding == 'one-hot':
        data = data_encoding(data, cols)
    data.to_csv(output_path, index=False, encoding='utf-8')

    return data