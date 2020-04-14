import pandas as pd

from overdue_preprocess import preprocess_qian
from overdue_preprocess import preprocess_zhong
from overdue_preprocess import preprocess_hou

from overdue_model import model_qian
from overdue_model import model_zhong
from overdue_model import model_hou

data = pd.read_csv('../../../Data/Auction_Management/Overdue_Estimate/raw/拍辅通静态表.csv')
data_output_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/'
data_qian_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/data_qian.csv'
data_zhong_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/data_zhong.csv'
data_hou_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/data_hou.csv'
model_path = '../../../output/Auction_Management/Overdue_Estimate/model/'

#拍卖前期数据预处理
preprocess_qian(data,data_output_path)

#拍卖中期数据预处理
preprocess_zhong(data,data_output_path)

#拍卖后期数据预处理
preprocess_hou(data,data_output_path)

#拍卖前期模型训练
model_qian(data_qian_path,model_path)

#拍卖中期模型训练
model_zhong(data_zhong_path,model_path)

#拍卖后期模型训练
model_hou(data_hou_path,model_path)

#拍卖前期模型分类
classify_qian(input_data,model_path)

#拍卖中期模型分类
classify_zhong(input_data,model_path)

#拍卖后期模型分类
classify_hou(input_data,model_path)
