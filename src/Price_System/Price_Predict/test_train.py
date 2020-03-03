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



# 模型训练
name_of_model = 'xgboost'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/xgboost_train_X.csv', index=False)

# 模型预测
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
data_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
result = predict_prediction(model_path=model_path,
                            data_path=data_path,
                            name_of_target='Final_Price')
print('xgboost finish')


# 模型训练
name_of_model = 'bayesianridge'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_bayesianridge.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/bayesianridge_train_X.csv', index=False)

# 模型预测
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_bayesianridge.pkl'
data_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
result = predict_prediction(model_path=model_path,
                            data_path=data_path,
                            name_of_target='Final_Price')
print('bayesianridge finish')


# 模型训练
name_of_model = 'lasso'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_lasso.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/lasso_train_X.csv', index=False)

# 模型预测
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_lasso.pkl'
data_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
result = predict_prediction(model_path=model_path,
                            data_path=data_path,
                            name_of_target='Final_Price')
print('lasso finish')




# 模型训练
name_of_model = 'gradientboosting'
input_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_gradientboosting.pkl'
train, test_result, train_X, test_X, test_y =  predict_model(input_path=input_path,
                                                             model_path = model_path,
                                                             name_of_model=name_of_model,
                                                             name_of_target='Final_Price')
train_X.to_csv('../../../output/Price_System/Price_Predict/results/gradientboosting_train_X.csv', index=False)

# 模型预测
model_path = '../../../output/Price_System/Price_Predict/model/price_predict_gradientboosting.pkl'
data_path = '../../../Data/Price_System/Price_Predict/feature/house/used_house_data_feature.csv'
result = predict_prediction(model_path=model_path,
                            data_path=data_path,
                            name_of_target='Final_Price')
print('gradientboosting finish')

print('all finish')