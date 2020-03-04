# 手动特征工程处理部分
# 处理特定数据集上的特定问题以及常见的数据格式问题
from fucset import data_latlng

def handpreprocessing(data):

    data_pre = data.copy()

    data_pre['Location'] = data_pre['Region'] + \
                            data_pre['Road'] + \
                            data_pre['Community_Name']

    # 手动删除无意义特征
    cols = ['ID','Unit_Price','Listing_Price','Transaction_Time','Listing_Time','Region','Road','Community_Name','Num_Bedroom', 'Num_Hall']
    data_pre.drop(cols,axis=1,inplace=True)

    # 手动区分数值型和字符型变量
    cols = {}
    cols['num'] = ['Final_Price','Transaction_Cycle','Num_Price_Adjustment',
                    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
                    'Property_Rights_Time']
    cols['str'] = list(set(list(data_pre.columns)).difference(set(cols['num'])))

    # 经纬度转换
    print("==== 经纬度转换 ====")
    data_latlng_encode = data_latlng(data_pre, 'train', cols)

    return data_latlng_encode, cols