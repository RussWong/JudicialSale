'''
    2020/01/13
    Funclist:
        plot_missing:绘制缺失率统计图并标记缺失率超过阈值mr_th的特征
        plot_col_unique:绘制特征的独立变量数
        plot_numhist:绘制数值型变量的统计直方图
        plot_strbar:绘制分类型变量的统计条形图
        plot_correlation:绘制变量相关性的热力图
'''

import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from scipy.stats import norm

def plot_missing(data,path,filename,mr_th=0.4):
    Num_all=data.shape[0]
    rate_series=pd.Series()
    for col in data.columns:
        tempdf=data[col]
        Num_missing=tempdf[tempdf.isnull()].shape[0]
        rate=Num_missing/Num_all
        rate_series[col]=rate
    rate_series=rate_series.rename('缺失率')
    fig=plt.figure(figsize=(30,25))
    data_plt=rate_series.sort_values(ascending=False)
    #     print(data_plt)
    x=data_plt.values*100
    y=data_plt.index
    x_inv=x[::-1]
    y_inv=y[::-1]
    x1=x_inv[x_inv>mr_th*100]
    x2=x_inv[x_inv<=mr_th*100]
    y1=y_inv[x_inv>mr_th*100]
    y2=y_inv[x_inv<=mr_th*100]

    b2=plt.barh(y2[x2>=0.5],x2[x2>=0.5])
    b1=plt.barh(y1[x1>0.5],x1[x1>0.5])#由于正序输出会从小到大地自高向低排列，所以需要倒序
#     i=0
    for rect in b1:
        w=rect.get_width()#获得柱状图的标签取值（即横向宽度，纵向应该获取高度）
        plt.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%(w),ha='left',va='center',fontsize=25)
#         i+=1
    for rect in b2:
        w=rect.get_width()#获得柱状图的标签取值（即横向宽度，纵向应该获取高度）
        plt.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%(w),ha='left',va='center',fontsize=25)
#         i+=1
    plt.xlim([0,110])
    plt.yticks(ticks=y_inv[x_inv>=0.5],fontsize=25)
    plt.xticks(fontsize=25)
    plt.xlabel('缺失率(%)',fontsize=25)
    plt.title('缺失率情况统计',fontsize=25)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
#     plt.show()
def plot_col_unique(data,path,filename):
    Num_all=data.shape[0]
    rate_series=pd.Series()
    for col in data.columns:
        tempdf=data[col]
        Num_unique=tempdf.unique().shape[0]
        rate_series[col]=Num_unique
    rate_series=rate_series.rename('独立值个数')
    #     print(rate_series)
    fig=plt.figure(figsize=(30,25))
    data_plt=rate_series.sort_values(ascending=False)
    #     print(data_plt)
    x=data_plt.values
    y=data_plt.index
    x_inv=x[::-1]
    y_inv=y[::-1]
    b=plt.barh(y_inv,x_inv)#由于正序输出会从小到大地自高向低排列，所以需要倒序
    i=0
    for rect in b:
        w=rect.get_width()#获得柱状图的标签取值（即横向宽度，纵向应该获取高度）
        plt.text(w,rect.get_y()+rect.get_height()/2,'%d'%(w),ha='left',va='center',fontsize=25)
        i+=1
    plt.xlim([0,x.max()*1.1])
    plt.yticks(ticks=y_inv,fontsize=25)
    plt.xticks(fontsize=25)
    plt.xlabel('Unique值个数',fontsize=25)
    plt.title('特征Unique值统计',fontsize=25)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
#     plt.show()

def plot_numhist(data,path,filename):
    Num_all=data.columns.shape[0]
    plt.figure(figsize=(20,Num_all*20))
    for i in range(Num_all):
        plt.subplot(Num_all,1,i+1)
        sns.distplot(data.iloc[:,i].dropna(),hist=True,kde=False,norm_hist=False,rug=True,vertical=False,label='distplot',fit=norm)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('取值',fontsize=25)
        plt.title('特征'+data.columns[i]+'数据分布',fontsize=25)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
#     plt.show()

def plot_strbar(data,path,filename):
    Num_all=data.columns.shape[0]
    plt.figure(figsize=(20,Num_all*20))
    for i in range(Num_all):
        plt.subplot(Num_all,1,i+1)
        str_counts=data.iloc[:,i].value_counts()
        if len(str_counts)>15:
            str_counts=str_counts.sort_values(ascending=False).iloc[0:15]
        plt.bar(str_counts.index,str_counts.values)
        plt.xticks(rotation=90,fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('取值',fontsize=25)
        plt.title('特征'+data.columns[i]+'主要数据分布',fontsize=25)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
#     plt.show()
    
def plot_correlation(data,path,filename):
    plt.subplots(figsize=(25, 30))
    sns.heatmap(data.corr(), annot=True, vmax=1, square=True,cmap='Blues',annot_kws={'size':15,'weight':'bold', 'color':'black'},fmt=".2f")
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    plt.xticks(rotation=90,fontsize=25)
    plt.yticks(rotation=0,fontsize=25)
    plt.xlabel('特征名称',fontsize=25)
    plt.title('特征相关度',fontsize=25)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
#     plt.show()