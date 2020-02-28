import math
import copy
import time
from impyute.imputation.cs import central_tendency, em
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
# TODO 缺失情况统计
def missing_rate(data,output_flag=False):
    """
    :input:data:df型输入数据，output_flag:bool型，是否print缺失率信息报告
    """
    Num_all=data.shape[0]
    res_dict={}
    if output_flag:
        print('========================缺失率统计========================')
    for col in data.columns:
        tempdf=data[col]
        Num_missing=tempdf[tempdf.isnull()].shape[0]
        rate=Num_missing/Num_all
        res_dict[col]=rate
        if output_flag:
            print('特征 %s 缺失率： %.2f %%'%(col,rate*100))
        if output_flag:
            print('==========================================================')
    return pd.Series(res_dict).rename('missing_rate')

# TODO 缺失值填充
def impencode(data,encoder):
    '''
    :function:编码非数值型变量
    :input:data为单栏的数据/Series,encoder为所选择的编码器
    '''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

import os
from sklearn.preprocessing import OrdinalEncoder
# TODO 缺失情况统计
def missing_rate(data,output_flag=False):
    """
    :input:data:df型输入数据，output_flag:bool型，是否print缺失率信息报告
    """
    Num_all=data.shape[0]
    res_dict={}
    if output_flag:
        print('========================缺失率统计========================')
    for col in data.columns:
        tempdf=data[col]
        Num_missing=tempdf[tempdf.isnull()].shape[0]
        rate=Num_missing/Num_all
        res_dict[col]=rate
        if output_flag:
            print('特征 %s 缺失率： %.2f %%'%(col,rate*100))
        if output_flag:
            print('==========================================================')
    return pd.Series(res_dict).rename('missing_rate')

# TODO 缺失值填充
def impencode(data,encoder):
    '''
    :function:编码非数值型变量
    :input:data为单栏的数据/Series,encoder为所选择的编码器
    '''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

def missing_value_processing(data,cols,mr_th=0.4):
    """
    先去除缺失率高于阈值mr_th的特征
    对num特征进行EM算法插补
    对str特征编码后EM算法插补再取整后反编码
    param:cols,栏标签（NUM表示数值型,STR表示非数值型）mr_th为缺失率阈值，高于该阈值的属性将被直接丢弃
    return:res插值后的数据dataframe
    """
    
    res=copy.deepcopy(data)
    mr=missing_rate(data)
    cols_del=mr[mr>mr_th].index.tolist()
    res.drop(cols_del,axis=1,inplace=True)
    if len(cols_del)>0:
        print('去除了缺失率超过40 % 的特征:')
        for name in cols_del:
            print(name)
    cols['num'] = [x for x in cols['num'] if x not in cols_del]
    cols['str'] = [x for x in cols['str'] if x not in cols_del]
    NUM_data=res[cols['num']]
    STR_data=res[cols['str']]
#     max_data_len=min(max_subset_num,math.round(NUM_data.shape[0]*0.3))
#     train_num_data=find_similar_set(NUM_data,max_num_len)
    print('数值型数据EM插补开始,总数据量：%d'%NUM_data.shape[0])
    start=time.time()
    em_imp_num = em(NUM_data.values)
    em_imp_data1 = pd.DataFrame(em_imp_num,columns=NUM_data.columns,index=NUM_data.index)
    finish=time.time()
    print('数值型数据EM插补完成，耗时%d秒'%(finish-start))
    res[cols['num']]=em_imp_data1
    print('非数值型数据EM插补开始,总数据量：%d'%STR_data.shape[0])
    enc = OrdinalEncoder()
    STR_data_placeholder=copy.deepcopy(STR_data)
    for col_name in STR_data_placeholder.columns:
        STR_data_placeholder[col_name].fillna(STR_data_placeholder[col_name].mode()[0],inplace=True)
    enc.fit(STR_data_placeholder)
    enc_col=OrdinalEncoder()
    for colname in STR_data.columns:
        impencode(STR_data[colname],enc_col)
    print('分类变量预编码完成，开始数据插值')
    start=time.time()
    em_imp_str = em(STR_data.astype(float).values)
    em_imp_str = np.round(em_imp_str)
    #有可能因为round导致encoder索引越界，因此需要作截断处理，此处的截断基于顺序编码
    for y in range(em_imp_str.shape[1]):
        maxlabel=enc.categories_[y].shape[0]-1
        minlabel=0
        for x in range(em_imp_str.shape[0]):
            if em_imp_str[x,y]>maxlabel:em_imp_str[x,y]=maxlabel
            if em_imp_str[x,y]<minlabel:em_imp_str[x,y]=minlabel
    STR_data_imp=enc.inverse_transform(em_imp_str)
    em_imp_data2=pd.DataFrame(STR_data_imp,columns=STR_data.columns,index=STR_data.index)
    res[cols['str']]=em_imp_data2
    finish=time.time()
    print('分类变量插值与反编码完成，共耗时%d秒'%(finish-start))        
    return res