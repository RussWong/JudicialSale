import pandas as pd
import numpy as np
import joblib

# data_path 输入csv的路径
# model_path 输入模型pkl的路径
# name_of_target 预测目标名称
def predict_prediction(model_path, data_path, name_of_target='Final_Price'):
    predictor = joblib.load(model_path)
    data = pd.read_csv(data_path)
    result = predictor.predict(data.drop([name_of_target], axis=1))

    return result




