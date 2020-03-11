import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from analysis import globalSurrogate,shap_compute,residuals_plot
# model_path 输入模型pkl的路径
# name_of_model 输入模型的名字
# data_path 训练数据集csv的路径
# output_path 输出图片的路径 '_shap_impact.png'
def predict_analysis(model_path, name_of_model, data_path, output_path):
    model = joblib.load(model_path)
    X_train = pd.read_csv(data_path)

    index, shap_mean_df = shap_compute(X_train, model, name_of_model)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.barh(index, shap_mean_df['shap_mean_values'].values)
    ax.set_yticks(index)
    ax.set_yticklabels(shap_mean_df['columns_name'].values)
    ax.set_xlabel('shaply值参数重要性', size=15)

    plt.savefig(output_path)

    return 