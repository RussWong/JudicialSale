import pandas as pd
import numpy as np
from handpreprocessing import handpreprocessing

# input_path 输入csv的路径
# output_path 输出csv的路径
# is_need 是否需要进行手动特征工程

def predict_hand(input_path, output_path, is_need=0):

    data = pd.read_csv(input_path)
    cols_num = ['Final_Price','Transaction_Cycle','Num_Price_Adjustment',
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time'] # 数值型变量名

    if is_need == 1:
        hand = handpreprocessing(data=data, cols_num=cols_num)
        data, cols = hand.run()
    else:
        cols = {}
        cols['num'] = cols_num
        cols['str'] = list(set(list(data.columns)).difference(set(cols['num'])))

    data.to_csv(output_path, encoding='utf-8', index=False)

    # cols 输入csv的标签分类信息{'num':[],'str':[]}
    return data, cols