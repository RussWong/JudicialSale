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


    X_train.rename(columns={'House_Type':'房型',
                            'Transaction_Cycle':'交易周期',
                            'Num_Look':'带看次数',
                            'Attention':'关注人数',
                            'Construction_Area':'建筑面积',
                            'Age':'建成年份',
                            'Renovation':'装修情况',
                            'Construction_struct':'建筑结构',
                            'Ladder_Ratio':'梯户比例',
                            'Elevator':'有无电梯',
                            'Storey':'所在楼层',
                            'Ladder':'电梯数',
                            'Household':'每层几户',
                            'Longitude':'经度',
                            'Latitude':'纬度'},
                   inplace=True)
    

    index, shap_mean_df = shap_compute(X_train, model, name_of_model)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.barh(index, shap_mean_df['shap_mean_values'].values)
    ax.set_yticks(index)
    ax.set_yticklabels(shap_mean_df['columns_name'].values)
    
    ax.set_xlabel('shaply值参数重要性', size=15)


    plt.savefig(output_path)

    return 