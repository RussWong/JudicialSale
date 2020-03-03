import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
sys.path.append('./module/')

from predict_preprocess import predict_preprocess
from predict_qualified import predict_qualified
from predict_encoding import predict_encoding
from predict_feature import predict_feature
from predict_model import predict_model
from predict_prediction import predict_prediction
from predict_analysis import predict_analysis

# 手工处理
input_path = '../../../Data/Price_System/Price_Predict/raw/house/used_house_data.csv'
output_path = '../../../Data/Price_System/Price_Predict/preprocess/house/used_house_data_preprocess.csv'
data_preprocess, cols = predict_preprocess(input_path=input_path,
                                           output_path=output_path,
                                           is_need=1)

# 数据清洗
mpl.rcParams['font.sans-serif'] = ['SimHei']
input_path =  '../../../Data/Price_System/Price_Predict/preprocess/house/used_house_data_preprocess.csv'
output_path = '../../../Data/Price_System/Price_Predict/qualified/house/used_house_data_qualified.csv'
docs_path = '../../../docs/Price_System/Price_Predict/used_house_data/'
data_qualified = predict_qualified(input_path=input_path,
                                   output_path=output_path,
                                   docs_path=docs_path,
                                   name_of_target='Final_Price', cols=cols)

# 编码
input_path='../../../Data/Price_System/Price_Predict/qualified/house/used_house_data_qualified.csv'
output_path='../../../Data/Price_System/Price_Predict/encoding/house/used_house_data_encoding.csv'
data_encoding = predict_encoding(input_path=input_path,
                                 output_path=output_path,
                                 cols=cols,
                                 name_of_target='Final_Price',
                                 type_of_encoding='target')

# 特征选择
input_path='../../../Data/Price_System/Price_Predict/encoding/house/used_house_data_encoding.csv'
output_path='../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
data_feature = predict_feature(input_path=input_path,
                               output_path=output_path,
                               num_of_feature=15, name_of_target='Final_Price')



# 模型训练bagging
name_of_model = 'bagging'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_bagging.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/bagging_train_X.csv', index=False)
print('bagging finish')


# 模型训练xgboost
name_of_model = 'xgboost'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/xgboost_train_X.csv', index=False)
print('xgboost finish')


# 模型训练bayesianridge
name_of_model = 'bayesianridge'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_bayesianridge.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/bayesianridge_train_X.csv', index=False)
print('bayesianridge finish')


# 模型训练lasso
name_of_model = 'lasso'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_lasso.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/lasso_train_X.csv', index=False)
print('lasso finish')


# 模型训练gradientboosting
name_of_model = 'gradientboosting'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_gradientboosting.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/gradientboosting_train_X.csv', index=False)
print('gradientboosting finish')


print('model train finish')



# # 模型预测bagging
# model_path = '../../../output/Price_System/Price_Predict/model/price_predict_bagging.pkl'
# data_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
# result = predict_prediction(model_path=model_path,
#                             data_path=data_path,
#                             name_of_target='Final_Price')

# # 模型分析bagging
# name_of_model = 'bagging'
# model_path = '../../../output/Price_System/Price_Predict/model/price_predict_bagging.pkl'
# data_path = '../../../output/Price_System/Price_Predict/results/bagging_train_X.csv'
# output_path = '../../../output/Price_System/Price_Predict/analysis/bagging_shap.png'
# predict_analysis(model_path=model_path,
#                  name_of_model=name_of_model,
#                  data_path=data_path,
#                  output_path=output_path)


# # 模型预测xgboost
# model_path = '../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
# data_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
# result = predict_prediction(model_path=model_path,
#                             data_path=data_path,
#                             name_of_target='Final_Price')

# # 模型分析xgboost
# name_of_model = 'xgboost'
# model_path = '../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
# data_path = '../../../output/Price_System/Price_Predict/results/xgboost_train_X.csv'
# output_path = '../../../output/Price_System/Price_Predict/analysis/xgboost_shap.png'
# predict_analysis(model_path=model_path,
#                  name_of_model=name_of_model,
#                  data_path=data_path,
#                  output_path=output_path)