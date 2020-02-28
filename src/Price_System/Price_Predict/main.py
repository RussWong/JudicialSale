import sys
import pandas as pd
import numpy as np
sys.path.append('./module/')

from predict_hand import predict_hand
from predict_qualified import predict_qualified
from predict_encoding import predict_encoding
from predict_feature import predict_feature
from predict_model import predict_model
from predict_prediction import predict_prediction
from predict_analysis import predict_analysis

# 手工处理
input_path = '../../../Data/Price_System/Price_Predict/raw/house/used_house_data_test.csv'
output_path = '../../../Data/Price_System/Price_Predict/intermediate/used_house_data_test_intermediate.csv'
data_hand, cols = predict_hand(input_path=input_path, output_path=output_path)

# 数据清洗
input_path =  '../../../Data/Price_System/Price_Predict/intermediate/used_house_data_test_intermediate.csv'
output_path = '../../../Data/Price_System/Price_Predict/qualified/used_house_data_test_qualified.csv'
docs_path = '../../../docs/Price_System/Price_Predict/'
data_qualified = predict_qualified(input_path=input_path,
                                   output_path=output_path,
                                   docs_path=docs_path,
                                   name_of_target='Final_Price', cols=cols)

# 编码
input_path='../../../Data/Price_System/Price_Predict/qualified/used_house_data_test_qualified.csv'
output_path='../../../Data/Price_System/Price_Predict/encoding/used_house_data_test_encoding.csv'
data_encoding = predict_encoding(input_path=input_path,
                                 output_path=output_path,
                                 cols=cols,
                                 name_of_target='Final_Price',
                                 type_of_encoding='target')

# 特征选择
input_path='../../../Data/Price_System/Price_Predict/encoding/used_house_data_test_encoding.csv'
output_path='../../../Data/Price_System/Price_Predict/feature/used_house_data_test_feature.csv'
data_feature = predict_feature(input_path=input_path,
                               output_path=output_path,
                               num_of_feature=15, name_of_target='Final_Price')

# 模型训练
input_path = '../../../Data/Price_System/Price_Predict/feature/used_house_data_test_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model='xgboost',
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/model/train_X.csv')

# 模型预测
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
data_path = '../../../Data/Price_System/Price_Predict/feature/used_house_data_test_feature.csv'
result = predict_prediction(model_path=model_path,
                            data_path=data_path,
                            name_of_target='Final_Price')