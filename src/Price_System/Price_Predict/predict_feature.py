import pandas as pd
import numpy as np

import warnings
import json
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor

# input_path 输入csv的路径
# output_path 输出csv的路径
# num_of_feature 特征数量
# name_of_target 预测目标名称
# feature_path 输出选择的特征路径
# cols 输入列信息
def predict_feature(input_path, output_path, feature_path, cols, num_of_feature=15, name_of_target='Final_Price'):

    warnings.filterwarnings("ignore")
    data = pd.read_csv(input_path)

    # x除去价格列的数据，y价格列；x_feature特征list
    y = data[name_of_target].values
    x = data.drop([name_of_target], axis=1).values
    x_feature = list(data.columns)
    x_feature.remove(name_of_target)

    # 方法一：单变量特征选择 f_regression 基于方差
    f_test, _ = f_regression(x, y)
    result_f_regression = list(zip([i for i in range(len(x_feature))], f_test))
    result_f_regression.sort(key=lambda x: x[1], reverse=True)
    fr_ranking = [0] * len(x_feature)
    rank = 1
    for i in result_f_regression:
        fr_ranking[i[0]] = rank
        rank += 1

    # 方法二：单变量特征选择 mutual_info_regression 基于互信息
    mutual_test = mutual_info_regression(x, y)
    result_mutual_info_regression = list(zip([i for i in range(len(x_feature))], mutual_test))
    result_mutual_info_regression.sort(key=lambda x: x[1], reverse=True)
    mu_ranking = [0] * len(x_feature)
    rank = 1
    for i in result_mutual_info_regression:
        mu_ranking[i[0]] = rank
        rank += 1

    # RFE的稳定性很大程度上取决于迭代时，底层用的哪种模型。
    # 没有经过正则化的回归是不稳定的，比如普通的回归（LR），那么RFE就是不稳定的。
    # 假如采用的是Lasso/Ridge，正则化的回归是稳定的，那么RFE就是稳定的。

    # 方法三：基于RFE方法，评估器使用LR线性回归
    rfe0 = RFE(estimator=LogisticRegression(), n_features_to_select=1)
    rfe0.fit(x, y)

    # 方法四：基于RFE方法，评估器使用Ridge回归,CV表示交叉验证
    rcv = RidgeCV([1, 2, 5, 10, 20, 100])
    rfe1 = RFE(estimator=rcv, n_features_to_select=1)
    rfe1.fit(x, y)

    # 方法五：基于RFE方法，评估器使用Lasso回归,CV表示交叉验证
    rfe2 = RFE(estimator=LassoCV(), n_features_to_select=1)
    rfe2.fit(x, y)

    # 方法六：基于RFE方法，评估器使用随机森林RandomForestRegressor
    rfe3 = RFE(estimator=RandomForestRegressor(), n_features_to_select=1)
    rfe3.fit(x, y)

    list_name = [fr_ranking, mu_ranking, list(rfe0.ranking_), list(rfe1.ranking_), list(rfe2.ranking_), list(rfe3.ranking_)]
    list_index = ['f_regression', 'mutual_info_regression', 'RFE_LR', 'RFE_RidgeCV', 'RFE_LassoCV', 'RFE_RandomForestRegressor']
    feature_select = pd.DataFrame(list_name, index=list_index, columns=x_feature)

    l = []  # 删除特征
    for i in x_feature:
        tmp1 = feature_select[i].map(lambda x: x > num_of_feature).sum()
        tmp2 = feature_select[i].sum()
        l.append([i, tmp1 + tmp2 / 100])
    l.sort(key=lambda x: x[1])
    for i in l[num_of_feature:]:
        data.drop(i[0], axis=1, inplace=True)

    for key in cols.keys():
        for feature in cols[key]:
            if feature not in l[0:num_of_feature]:
                cols[key].remove(feature)

    fw = open(feature_path, 'w', encoding='UTF-8')
    json.dump(cols, fw)
    fw.close()
    # fw = open(feature_path, 'w', encoding='UTF-8')
    # for i in l[0:num_of_feature]:
    #     fw.write(i[0] + '\n')
    # fw.close()
    data.to_csv(output_path, index=False, encoding='utf-8')

    return data