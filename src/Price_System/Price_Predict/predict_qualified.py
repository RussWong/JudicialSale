import pandas as pd
import numpy as np
from fucset import data_duplicate, data_anomaly
from data_overview import plot_missing, plot_col_unique, plot_numhist, plot_strbar, plot_correlation
from missing import missing_value_processing
import warnings

# input_path 输入csv的路径
# output_path 输出csv的路径
# docs_path 输出文档的路径
# cols 输入csv的标签分类信息{'num':[],'str':[]}
# name_of_target 预测目标名称

def predict_qualified(input_path, output_path, cols, docs_path='../docs/', name_of_target='Final_Price'):
    raw_data = pd.read_csv(input_path)

    # TODO 1.1 数据概述
    warnings.filterwarnings("ignore")
    plot_missing(raw_data, docs_path, 'missing_rate')
    plot_col_unique(raw_data, docs_path, 'unique_count')
    plot_numhist(raw_data[cols['num']].dropna(axis=1), docs_path, 'num_varb_distribution')
    plot_strbar(raw_data[cols['str']].dropna(axis=1), docs_path, 'str_varb_distribution')
    plot_correlation(raw_data.dropna(axis=1), docs_path, 'corelation_heatmap')

    # TDO 1.2 数据去重
    data_duplicate(raw_data)

    # TODO 1.3 缺失值处理
    missing_pre = missing_value_processing(raw_data, cols)
    for key in cols.keys():
        for x in cols[key]:
            if x not in missing_pre.columns:
                cols[key].remove(x)

    # 异常检测
    data = data_anomaly(missing_pre, cols, name_of_target, docs_path)

    data.to_csv(output_path, index=False, encoding='utf-8')

    return data