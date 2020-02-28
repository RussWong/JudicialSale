import numpy as np
import pandas as pd
from train import Model_train
import joblib

# input_path 输入csv的路径
# model_path 输出pkl的路径
# name_of_target 预测目标名称
# 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor
# 可选参数 bayesianridge, lasso, gradientboosting, bagging, xgboost
def predict_model(input_path, model_path, name_of_model='xgboost', name_of_target='Final_Price'):
    data = pd.read_csv(input_path)
    model_instance = Model_train(data, name_of_target, name_of_model=name_of_model)
    if name_of_model == 'bayesianridge':
        train, test_result, train_X, test_X, test_y = model_instance.BRreg()
    elif name_of_model == 'lasso':
        train, test_result, train_X, test_X, test_y = model_instance.Lreg()
    elif name_of_model == 'gradientboosting':
        train, test_result, train_X, test_X, test_y = model_instance.GBreg()
    elif name_of_model == 'bagging':
        train, test_result, train_X, test_X, test_y = model_instance.Breg()
    elif name_of_model == 'xgboost':
        train, test_result, train_X, test_X, test_y = model_instance.XGBreg()
    else:
        pass

    joblib.dump(train, model_path)
    # for key, values in test_result.items():
    #     f.write(key + '  ' + str(round(values, 3)) + '\n')
    # train_X.to_csv(f ,index=False)
    # test_X.to_csv(f ,index=False)
    # test_y.to_csv(f ,index=False)
    return train, test_result, train_X, test_X, test_y



