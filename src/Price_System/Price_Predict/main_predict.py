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
# standModel _path输入标准化模型stand.pkl路径
# model_path 预测模型的path
# select_feature_path 特征选择文件路径
# column_stand_path 编码txt文件路径
# cols_path
def price_predict(raw_data, encoder_path, standModel_path, model_path, select_feature_path, column_stand_path, cols_path):

    # 读取文件，获得select_feature
    fr = open(select_feature_path, 'r', encoding='UTF-8')
    select_feature = []
    for line in fr:
        select_feature.append(line.strip())       
    fr.close()
    
    # 读取文件，获得column_stand
    fr = open(column_stand_path, 'r', encoding='UTF-8')
    column_stand = []
    for line in fr:
        column_stand.append(line.strip())       
    fr.close()

    # 获取需要编码的str类型特征  cols_str  
    fr = open(cols_path, 'r', encoding='UTF-8')
    cols = json.load(fr)       
    fr.close()
    cols_str = []
    for i in cols['str']:
        if i in select_feature:
            cols_str.append(i)
  
    # 输入数据缺失检查
    data = raw_data.copy()
    data_err = []
    for i in range(len(data.isnull().sum(1))):
        if data.isnull().sum(1)[i] != 0:
            data_err.append(i)
    if data_err:
        print('以下输入数据存在缺失:')
        print('第' + str(data_err)[1:-1] + '条')
        return [-1]
        # drop行-------------------------------------!------

    # 经纬度编码
    data['Location'] = data['Region'] + data['Road'] + data.copy()['Community_Name']
    data.drop(['Region', 'Road', 'Community_Name'], axis=1, inplace=True)
    data_lat = data_latlng(data)

    # 数据编码
    encoder = pd.read_csv(encoder_path, index_col=0)
    data_encoding = data_encoding_predict(data_lat, cols_str, encoder)

    # 编码失败检测
    encode_err = []
    for i in range(len(data_encoding.isnull().sum(1))):
        if data_encoding.isnull().sum(1)[i] != 0:
            encode_err.append(i)
    if encode_err:
        print('以下输入数据编码失败:')
        print('第' + str(encode_err)[1:-1] + '条')
        return [-1]
        # drop行-------------------------------------!------

    # 数据标准化
    data_norm = predict_standardization(data=data_encoding,
                                         mode='predict',
                                         target='Final_Price',
                                         path_model=standModel_path,
                                         column_stand=column_stand,
                                         select_feature=select_feature)

    # 预测
    predictor = joblib.load(model_path)
    result = predictor.predict(data_norm)
    result = list(result)


    return result


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


def data_encoding_predict(raw_data, cols_str, encoder):
    data = raw_data.copy()
    for i in cols_str:
        data[i] = data[i].map(encoder['Value'])

    return data


def predict_standardization(data, mode, target, path_model, column_stand, select_feature=None):
    if mode == 'train':
        data_trans = data_standardization(data, target, path_model, 'train')

        fw = open(path_stand_feature, 'w', encoding='UTF-8')
        for i in list(data_trans.columns):
            fw.write(i + '\n')
        fw.close()

    elif mode == 'predict':
        column_add = list(set(list(column_stand)).difference(set(select_feature)))

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
        print('remove target')

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

