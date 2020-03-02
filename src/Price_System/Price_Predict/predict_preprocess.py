import pandas as pd
import numpy as np
from handpreprocessing import handpreprocessing

# input_path 输入csv的路径
# output_path 输出csv的路径
# is_need 是否需要进行手动特征工程

def predict_preprocess(input_path, output_path, is_need=1):

    data = pd.read_csv(input_path)

    if is_need == 1:
        data, cols = handpreprocessing(data=data)
    else:
        data_types = data.dtypes
        cols_num = [data_types.index[i] for i in range(len(data_types)) if data_types[i] in ['int', 'float']]

        cols = {}
        cols['num'] = cols_num
        cols['str'] = list(set(list(data.columns)).difference(set(cols['num'])))

    data.to_csv(output_path, encoding='utf-8', index=False)

    # cols 输入csv的标签分类信息{'num':[],'str':[]}
    return data, cols