

'''
    Date: 2020/3/7
    Func_list:
    Version：    2.1.2
    Info：
        V_2.1.1：（完成测试）
                修改编码部分代码，注释onehot编码，修改target编码，添加预测功能。
        V_2.1.2: （完成测试）
                修改标准化部分代码，返回数据使之能保存
                数据标准化移到编码之后，所以处理都为数值型，故不需要col，但需target
                保存标准化模型

'''


import time
import pandas as pd
import numpy as np
import os

# 获取经纬度
import requests
from bs4 import BeautifulSoup
import json
# 分类与数值的编码
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
import joblib

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def data_duplicate(data):
    print('==== 数据去重开始 ====')
    data.drop_duplicates(keep='first', inplace=True)
    data.reset_index(drop=True,inplace=True)
    print('去重后数据规格：'+str(data.shape))
    print('==== 数据去重结束 ====')


def data_latlng(data,ways,cols,path=''):
    '''
        Function: 
            Transform to latitude and longitude.
        Input:
            data:       type=DataFrame, it's data[['Location']], 传不了Series
            ways:       type=string, values=['predict','train']

            
    '''
    print('==== 经纬度转换开始 ====')
    start = time.time()
    # 经纬度转换
    for i in range(data.shape[0]):
        if(ways == 'train'):
            if(i>0 and i%5000 == 0):
                print('Coding number is %d' %i)
        if(ways == 'predict' and i>0 and i%10 == 0):
            print('Coding number is %d' %i)
        if(data.loc[i,'Location'] is np.nan):      # 值为NaN
            tmp = {}
            tmp['lng'] = np.nan
            tmp['lat'] = np.nan
        else:
            tmp=Latlng(data.loc[i,'Location'],i)
        data.loc[i,'Longitude'] = tmp['lng']
        data.loc[i,'Latitude'] = tmp['lat']
    # 数据变更
    data.drop('Location',axis=1,inplace=True)
    cols['str'].remove('Location')
    cols['num'].append('Longitude')
    cols['num'].append('Latitude')
    # 保存
    # data.to_csv(path+'data_latlng.csv',index=False)
    end = time.time()
    print('转换后数据规格：'+str(data.shape))
    print('耗时：%.4f' %(end-start))
    print('==== 经纬度转换结束 ====')
    return data



def Latlng(location,i):
    '''
        Function:
            Transformation from location to latitude and longitude.
        Input:
            location: type=string
        Output:
            latlng:   type=dict.

    '''
    tmp='上海市'+location
    url= 'http://api.map.baidu.com/geocoder?address='+tmp+'&output=json&key=f247cdb592eb43ebac6ccd27f796e2d2'
    try:
        html = requests.get(url,headers={'Connection':'close'})
        json1 = BeautifulSoup(html.text,'html.parser')
        if len(json1.text) != 0:
            json1 = json.loads(json1.text)
            lat = json1['result']['location']['lat']
            lng = json1['result']['location']['lng']
        else:
            lat = np.nan
            lng = np.nan
    except Exception as e:
        # print(' 第 %d 出现错误，休眠重连，错误是：%s' %(i,e))
        time.sleep(10)
        json1=Latlng(location,i)
        lat=json1['lat']
        lng=json1['lng']
    return {'lat':lat,'lng':lng}



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
    else:
        print('错误：没有此目标标签')

    if mode == 'train':
        if method == 'Robust':
            scaler = RobustScaler()
        scaler.fit(data[columns])
        # store
        joblib.dump(scaler, path)
    elif mode =='predict':
        if os.path.exists(path):
            scaler = joblib.load(path)
        else:
            print('错误：未存在模型')


    data[columns] = pd.DataFrame(scaler.transform(data[columns]),columns=columns)
    
    print('==== 数据标准化结束 ====')

    return data


def data_anomaly(data_all,cols,ylabel,path):
    """
      param:
      function:
    """
    print('异常检测开始')
    start=time.time()

    data = data_all[cols['num']].copy()
    res = []

    for i in range(100):
        clf = IsolationForest(behaviour='new', contamination='auto')
        pred = clf.fit_predict(data)
        data['Outlier'] = pred

        if i == 0: 
            res= list(data[data['Outlier']==-1].index)
            data.drop('Outlier', axis=1, inplace=True)
            continue

        tmp = list((set(res) & set(list(data[data['Outlier']==-1].index))))

        if len(tmp) == len(res) or i == 99:
            print('Iteration num:',i)
            plt_anomaly(data,ylabel,path)
            break
        else:
            res = tmp
            data.drop('Outlier', axis=1, inplace=True)

    print('异常检测数：', len(res))
    data_all.drop(res, inplace=True)
    data_all.reset_index(drop=True, inplace=True)

    print('异常检测后数据规格：'+str(data.shape))
    end=time.time()
    print('耗时：%.4f' %(end-start))
    print('异常检测结束')
    return data_all 



def plt_anomaly(data,ylabel,path,name='IsolationForest'):
    '''
        data:   数值型数据
        (每个数据集需微调)
    '''
    print('=========异常点作图========')
    num = len(data.columns) - 1
    plt.figure(figsize=(6.4,num*4.8))

    for i in range(num):
        plt.subplot(num,1,i+1,title='Figure '+str(i+1))
        plt.xlabel(list(data.columns)[i])
        plt.ylabel(ylabel)
        plt.scatter(data.iloc[:,i],data.loc[:,ylabel],s=0.4,c=data['Outlier'])
        
    plt.subplots_adjust(top=0.965)
    plt.suptitle(name)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path+name+'.png')
    plt.show()


# def data_encoding(data,cols):
#     """
#       param:
#       function:
#     """

#     print('数据编码开始')
#     start=time.time()

#     if 'str' in cols:
#         onehot = OneHotEncoder()
#         #tmp_matrix = onehot.fit_transform(data[cols['str']]).toarray()
#         onehot.fit(data[cols['str']])
#         tmp_matrix=onehot.transform(data[cols['str']]).toarray()
#         data_category = pd.DataFrame(tmp_matrix,columns=onehot.get_feature_names())
#         data.drop(cols['str'], axis=1, inplace=True)
#         data = pd.concat([data,data_category], axis=1)
#         print('编码后数据规格：'+str(data.shape))

#     end=time.time()
#     print('耗时：%.4f' %(end-start))
#     print('数据编码结束')
#     return data


def data_encoding_predict(raw_data, cols, encoder):
    '''
        param:
            data: dataframe
            cols: list, column name of categorical data
            encoder: dataframe, to encode data
        Info:
            Target encoding for categorical data when predict
        Note:
            If data value is not in encoder, it will return Null 

    '''
    data = raw_data.copy()

    if cols is not None:
        for i in cols:
            data[i] = data[i].map(encoder['Value'])

    return data



    
def data_encoding_train(raw_data, cols, target, m=300):
    '''
        param:
            data: dataframe
            cols: list, column name of categorical data
            target: str, label of target
            m: int, the weight of overall mean
        Info:
            Target encoding for categorical data when train the model
            url: http://kodgv.xyz/2019/04/08/%E7%AB%9E%E8%B5%9B%E7%BB%8F%E9%AA%8C/targetencoding/

    '''

    print('==== 数据编码开始 ====')
    start = time.time()

    data = raw_data.copy()
    encoder = pd.DataFrame()

    if cols is not None:
        for i in cols:
            data.loc[:,i], encoder_tmp = target_encoding(data, i, target, m)
            encoder = pd.concat([encoder, encoder_tmp])

        encoder.columns = ['Value']
        print('编码后数据规格：' + str(data.shape))
        print('编码表数据规格：' + str(encoder.shape))

    end = time.time()
    print('耗时：%.4f' %(end - start))
    print('==== 数据编码结束 ====')
    return data, encoder


def target_encoding(data, group, target, m):
    # Compute the global mean
    mean = data[target].mean()

    # Compute the number of values and the mean of each group
    agg = data.groupby(group)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return data[group].map(smooth), smooth


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

        data_stand = pd.concat([data_predict, pd.DataFrame(columns=column_add)], axis=1)
        data_trans = data_standardization(data_stand[column_stand], target, path_model, 'predict')
        data_trans.drop(column_add, axis=1, inplace=True)

    return data_trans
















