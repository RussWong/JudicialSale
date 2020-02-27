# 手动特征工程处理部分
# 处理特定数据集上的特定问题以及常见的数据格式问题
from fucset import data_latlng

"""
目前支持的功能:

1.针对房产数据的地理位置编码(位置信息转换为经纬度)
2.无意义特征删除(根据先验知识判断)
3.特定问题处理接口

"""
class handpreprocessing():
    def __init__(self, data, type_of_dataset='House', cols_num=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']):
        """
        param:
        data : pandas.DataFrame
        type_of_dataset : str
        cols_num : list
        """
        super(handpreprocessing, self).__init__()
        self.data = data
        self.type_of_dataset = type_of_dataset
        self.cols_num = cols_num

    def run(self):

        # 地理编码
        if self.type_of_dataset == 'House':
            
            data_pre = self.data.copy()
            
            data_pre['Location'] = self.data.copy()['Region'] + self.data.copy()['Road'] + self.data.copy()['Community_Name']
            # 人工特征删除 
            # ! 这部分放到handpreprocessing_user中 
            cols = ['ID','Unit_Price','Listing_Price','Transaction_Time','Listing_Time','Region','Road','Community_Name','Num_Bedroom', 'Num_Hall']
            data_pre.drop(cols,axis=1,inplace=True)

            cols = {}
            cols['num'] = list(self.cols_num)
            cols['str'] = list(set(list(data_pre.columns)).difference(set(cols['num'])))
            # print(data_pre.loc[0, 'Location'])
            # print(cols['str'])
            print("经纬度转换")
            data_latlng_encode = data_latlng(data_pre, 'train', cols)
            return data_latlng_encode, cols
        else:
            cols = {}
            cols['num'] = list(self.cols_num)
            cols['str'] = list(set(list(data_pre.columns)).difference(set(cols['num'])))
            return self.data, cols

class handpreprocessing_user(handpreprocessing):
    def __init__(self, data, type_of_dataset='House'):
        super(handpreprocessing_user, self).__init__(data, type_of_dataset=type_of_dataset)
    
    pass

    



