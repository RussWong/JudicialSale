'''
    File: trend_main.py
    Date: 2020/4/4
    Author：herozen
    Version: v1.0


'''



from trend_preprocess import trend_preprocess
from trend_qualified import trend_qualified
from trend_dataset import trend_dataset
from trend_model import trend_model



data_input_path = '../../../Data/Price_System/Price_Predict/raw/house/used_house_data.csv'
parameter_path  = './innerdata/paramter.txt'
data_qualified_path = '../../../Data/Price_System/Price_Trend/qualified/house/used_house_data_qualified.csv'
dataset_path = '../../../Data/Price_System/Price_Trend/dataset/'
index_path = './innerdata/index.csv'

output_models_path = '../../../output/Price_System/Price_Trend/models/'
output_files_path = '../../../output/Price_System/Price_Trend/results/'


# 预处理
trend_preprocess(data_input_path, parameter_path)

# 数据处理（去重与缺失）
trend_qualified(data_input_path, data_qualified_path, parameter_path)

# 数据集生成
trend_dataset(data_qualified_path, dataset_path, parameter_path, index_path)

# 模型
trend_model(dataset_path, index_path, output_models_path, output_files_path)