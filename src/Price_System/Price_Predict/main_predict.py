import pandas as pd
import numpy as np
import requests
import json
import joblib
import os

from sklearn.preprocessing import RobustScaler
from bs4 import BeautifulSoup

# raw_data 输入预测的dataframe
# encoder_path 输入的编码文件csv路径
# path_standModel 输入标准化模型stand.pkl
# path_stand_feature 输入txt
# path_select_feature 输入选择的feature.json
def price_predict(raw_data, encoder_path, path_standModel, path_stand_feature, path_select_feature):
    data = raw_data.copy()
    # 缺失处理---------------------------------！！！---------------
#     data_err = []
#     for i in range(len(data.isnull().sum(1))):
#         if data.isnull().sum(1)[i] != 0:
#             data_err.append(i)

    fr = open(path_select_feature, 'r', encoding='UTF-8')
    cols = json.load(fr)
    fr.close()

    # 经纬度编码
    data['Location'] = data['Region'] + data['Road'] + data.copy()['Community_Name']
    cols_lat = ['Region', 'Road', 'Community_Name']
    data.drop(cols_lat, axis=1, inplace=True)
    data_lat = data_latlng(data)

    # 数据编码
    encoder = pd.read_csv(encoder_path, index_col=0)
    data_encoding = data_encoding_predict(data_lat, cols['str'], encoder)

    # 编码失败---------------------------!!!--------------------------------
#     encode_err = []
#     for i in range(len(data_encoding.isnull().sum(1))):
#         if data_encoding.isnull().sum(1)[i] != 0:
#             encode_err.append(i)

    # 数据标准化
    data_norm = predict_standardization(data=data_encoding,
                                         mode='predict',
                                         target='',
                                         path_model=path_standModel,
                                         path_stand_feature=path_stand_feature,
                                         path_select_feature=path_select_feature)



    # 预测
    # predictor = joblib.load(model_path)
    # result = predictor.predict(data_norm, axis=1))


    return data_norm


def data_latlng(data):
    for i in range(data.shape[0]):
        if data.loc[i, 'Location'] is np.nan:  # 值为NaN
            tmp = {}
            tmp['lng'] = np.nan
            tmp['lat'] = np.nan
        else:
            tmp = Latlng(data.loc[i, 'Location'], i)
        data.loc[i, 'Longitude'] = tmp['lng']
        data.loc[i, 'Latitude'] = tmp['lat']
    data.drop('Location', axis=1, inplace=True)
    return data


def Latlng(location, i):
    tmp='上海市'+location
    url= 'http://api.map.baidu.com/geocoder?address='+tmp+'&output=json&key=f247cdb592eb43ebac6ccd27f796e2d2'
    html = requests.get(url, headers={'Connection':'close'})
    json1 = BeautifulSoup(html.text, 'html.parser')
    if len(json1.text) != 0:
        json1 = json.loads(json1.text)
        lat = json1['result']['location']['lat']
        lng = json1['result']['location']['lng']
    else:
        lat = np.nan
        lng = np.nan
    return {'lat':lat, 'lng':lng}


def data_encoding_predict(raw_data, cols, encoder):
    data = raw_data.copy()
    if cols is not None and 'str' in cols:
        for i in cols['str']:
            data[i] = data[i].map(encoder['Value'])

    return data


def predict_standardization(data, mode, target, path_model, path_stand_feature, path_select_feature=None):
    if mode == 'train':
        data_trans = data_standardization(data, target, path_model, 'train')

        fw = open(path_stand_feature, 'w', encoding='UTF-8')
        for i in list(data_trans.columns):
            fw.write(i + '\n')
        fw.close()

    elif mode == 'predict':
        column_stand = []
        fr = open(path_stand_feature, 'r', encoding='UTF-8')
        for line in fr:
            line = line.strip()
            column_stand.append(line)
        fr.close()

        column_predict = []
        fr = open(path_select_feature, 'r', encoding='UTF-8')
        dic = json.load(fr)
        fr.close()
        for key in dic.keys():
            for fe in dic[key]:
                column_predict.append(fe)

        column_add = list(set(list(column_stand)).difference(set(column_predict)))

        data_stand = pd.concat([data, pd.DataFrame(columns=column_add)], axis=1)
        data_trans = data_standardization(data_stand[column_stand], target, path_model, 'predict')
        data_trans.drop(column_add, axis=1, inplace=True)

    return data_trans


def data_standardization(raw_data, target, path, mode='train', method='Robust'):
    """
    param:
        data: type=DataFrame, all data
        target: type=string, label of target
        path: type=string, model path
        method：type=string, method to scale data
    function:
        Standardize data
    Note:
        Need to distinguish standardization, normalization, Gaussian Mapping.
        Leave room for future improvement.

    """
    print('==== 数据标准化开始 ====')
    data = raw_data.copy()
    columns = list(data.columns)

    if target in columns:
        columns.remove(target)
#     else:
#         print('错误：没有此目标标签')

    if mode == 'train':
        if method == 'Robust':
            scaler = RobustScaler()
        scaler.fit(data[columns])
        # store
        joblib.dump(scaler, path)
    elif mode == 'predict':
        if os.path.exists(path):
            scaler = joblib.load(path)
        else:
            print('错误：未存在模型')

    data[columns] = pd.DataFrame(scaler.transform(data[columns]), columns=columns)

    print('==== 数据标准化结束 ====')

    return data

