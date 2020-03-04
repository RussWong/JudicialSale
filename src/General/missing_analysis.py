import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def missing_analysis(data_raw,threshould=0.5,ref_dict=None,
                     isNeed_plot=False,fig_size=(30,25),
                     path='./',filename='missing_rate_feature'):
    '''
    function: main function of the missing analysis
    :input:
        data_raw: input dataframe
        ref_dict: required values that desire to exist in the data

    :output:
        None
    '''
    data=copy.deepcopy(data_raw)
    print('**************************** missing analysis started ****************************')
    missing_cols=missing_null_col(data,threshould,isNeed_plot,fig_size,path,filename)
    if missing_cols.shape[0]>0:
        print('The feature(s):')
        for colname in missing_cols:
            print(colname)
        print('are highly missing,we recommand to delete the feature(s)')
    else:
        print('The missing rate of all the features is OK')
    missing_inds=missing_null_row(data,threshould)
    if missing_inds.shape[0]>0:
        print('The sample(s) with the index:')
        for ind in missing_inds:
            print(ind)
        print('are highly missing,we recommand to delete the sample(s)')
    else:
        print('The missing rate of all the samples is OK')
    if ref_dict:
        not_covered,extra= missing_ref_cover(data,ref_dict)
        if len(not_covered.keys())>0:
            print('Warning:Uncovered values in the required list!')
            for key in  not_covered.keys():
                print('The feature ',key,' doesn\'t cover the following values:')
                print(','.join(not_covered[key]))
        if len(extra.keys())>0:
            print('Warning:extra values not in the required list!')
            for key in  extra.keys():
                print('The feature ',key,' contains the following values:')
                print(','.join(extra[key]))

    print('**************************** missing analysis finished ****************************')

def missing_null_col(data_raw,threshould=0.5,isNeed_plot=False,fig_size=(30,25),path='./',filename='missing_rate_feature'):
    '''
    function: compute missing rate of the input  feature by feature
    :input:
        data_raw: input dataframe
        threshould: missing rate higher than this threshould value would be highlightened in the output plot
        isNeed_plot: whether a missing rate statistics plot is needed to be plotted (True if needed,otherwise False)
        fig_size: size of the output figure
        path: path to save the output figure
        filename: filename of the output figure
    :output:

        missing_rate_feature.png
    '''
    Num_all=data_raw.shape[0]
    rate_series=pd.Series()
    for col in data_raw.columns:
        tempdf=data[col]
        Num_missing=tempdf.isnull().sum(axis=0)
        rate=Num_missing/Num_all
        rate_series[col]=rate
    rate_series=rate_series
    if isNeed_plot:
        fig=plt.figure(figsize=fig_size)
        data_plt=rate_series.sort_values(ascending=False)
        x=data_plt.values*100
        y=data_plt.index
        x_inv=x[::-1]
        y_inv=y[::-1]
        x1=x_inv[x_inv>threshould*100]
        x2=x_inv[x_inv<=threshould*100]
        y1=y_inv[x_inv>threshould*100]
        y2=y_inv[x_inv<=threshould*100]
        b2=plt.barh(y2[x2>=0.5],x2[x2>=0.5])
        b1=plt.barh(y1[x1>0.5],x1[x1>0.5])#由于正序输出会从小到大地自高向低排列，所以需要倒序
        for rect in b1:
            w=rect.get_width()#获得柱状图的标签取值（即横向宽度，纵向应该获取高度）
            plt.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%(w),ha='left',va='center',fontsize=25)
        for rect in b2:
            w=rect.get_width()#获得柱状图的标签取值（即横向宽度，纵向应该获取高度）
            plt.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%(w),ha='left',va='center',fontsize=25)
        plt.xlim([0,110])
        plt.yticks(ticks=y_inv[x_inv>=0.5],fontsize=25)
        plt.xticks(fontsize=25)
        plt.xlabel('missing rate(%)',fontsize=25)
        plt.title('Feature Missing Rate Statistics ',fontsize=25)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.tight_layout()
        plt.savefig(path+filename+'.png')
        print('Feature missing rate plotted!')
    invalid_cols=rate_series[rate_series>threshould].index
    return invalid_cols

def missing_null_row(data_raw,threshould=0.5,path='./',filename='highly_missing_sample'):
    '''
    function: compute missing rate of the input  sample by sample
    :input:
        data_raw: input dataframe
        threshould: missing rate higher than this threshould value would be highlightened in the output plot
        path: path to save the highly missing data
        filename: filename of the output file
    :output:
        invalid_rows: indexs of samples having a missing rate(column-wise) higher than threshould
    '''
    Num_all=data_raw.shape[1]
    rate_series=pd.Series()
    for ind,__ in data_raw.iterrows():
        tempdf=data.loc[ind,:]
        Num_missing=tempdf.isnull().sum(axis=1)
        rate=Num_missing/Num_all
        rate_series.loc[ind]=rate
    invalid_rows=rate_series[rate_series>threshould].index
    invalid_data=data_raw.loc[invalid_rows,:]
    invalid_data.to_csv(path+filename+'.csv')
    return invalid_rows

def missing_ref_cover(data_raw,ref_dict):
    '''
    function: check the coverage of the reference values and extra values in the input data
    :input:
        data_raw: input dataframe
        ref_dict: required values that desire to exist in the data
    :output:
         not_covered:dict of the uncovered data given the keys in the ref_dict
         extra:dict of the extra data given the keys in the ref_dict
    '''
    not_covered={}
    extra={}
    for key in ref_dict.keys():
        if key not in (data_raw.columns):
            print('The feature %s not found in data'%key)
        not_covered.update({key:ref_dict[key]})
    for col in data_raw.columns:
        if col not in ref_dict.keys():
            extra.update({col:data_raw[col].unique().tolist()})
        else:
            raw=set(data_raw[col].unique().tolist())
            ref=set(ref_dict[col])
            extra.update({col:[raw-ref]})
            not_covered.update({col:[ref-raw]})
    return not_covered,extra