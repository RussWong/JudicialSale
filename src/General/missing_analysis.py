#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import datetime
from datetime import timedelta
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
class missing_analysis:
    def col_isnull(data_raw,threshould=0.5,isNeed_plot=False,fig_size=None,path='./',filename='missing_rate_feature'):
        '''Compute missing rate of the input  feature by feature

        Args:
            data_raw: input dataframe
            threshould: missing rate higher than this threshould value would be highlightened in the output plot
            isNeed_plot: whether a missing rate statistics plot is needed to be plotted (True if needed,otherwise False)
            fig_size: size of the output figure
            path: path to save the output figure
            filename: filename of the output figure

        Returns:
            invalid_cols: names of columns that has a missing rate above threshould
        '''
        data=copy.deepcopy(data_raw)
        Num_all=data.shape[0]
        rate_series=pd.Series()
        for col in data.columns:
            tempdf=data[col]
            Num_missing=tempdf.isnull().sum(axis=0)
            rate=Num_missing/Num_all
            rate_series[col]=rate
        rate_series=rate_series
        if isNeed_plot:
            if not fig_size:
                fig_size=(10,1*rate_series.shape[0])
            fig=plt.figure(figsize=fig_size)
            data_plt=rate_series.sort_values(ascending=True)
            x=data_plt.values*100
            y=data_plt.index
            x1=x[x>threshould*100]
            x2=x[x<=threshould*100]
            y1=y[x>threshould*100]
            y2=y[x<=threshould*100]
            b2=plt.barh(y2[x2>=0.5],x2[x2>=0.5])
            b1=plt.barh(y1[x1>0.5],x1[x1>0.5])
            for rect in b1:
                w=rect.get_width()#value of the missing rate (in barh-plot,the width represents the value)
                plt.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%(w),ha='left',va='center',fontsize=25)
            for rect in b2:
                w=rect.get_width()#value of the missing rate (in barh-plot,the width represents the value)
                plt.text(w,rect.get_y()+rect.get_height()/2,'%.2f'%(w),ha='left',va='center',fontsize=25)
            plt.xlim([0,110])
            plt.yticks(ticks=y[x>=0.5],fontsize=25)
            plt.xticks(fontsize=25)
            plt.xlabel('missing rate(%)',fontsize=25)
            plt.title('Feature Missing Rate Statistics ',fontsize=25)
            if not os.path.exists(path):
                os.mkdir(path)
            plt.tight_layout()
            plt.savefig(path+filename+'.png')
            print('Feature missing rate plotted!')
        invalid_cols=rate_series>threshould
        return invalid_cols
    def row_isnull(data_raw,threshould=0.5,path='./',filename='highly_missing_sample'):
        '''compute missing rate of the input  sample by sample
        Args:
            data_raw: input dataframe
            threshould: missing rate higher than this threshould value would be highlightened in the output plot
            path: path to save the highly missing data
            filename: filename of the output file
        Returns:
            invalid_rows: indexs of samples having a missing rate(column-wise) higher than threshould
        '''
        data=copy.deepcopy(data_raw)
        Num_all=data.shape[1]
        rate_series=pd.Series()
        for ind,__ in data.iterrows():
            tempdf=data.loc[ind,:]
            Num_missing=tempdf.isnull().sum()
            rate=Num_missing/Num_all
            rate_series.loc[ind]=rate
        invalid_rows=rate_series>threshould
        invalid_data=data.loc[invalid_rows,:]
        invalid_data.to_csv(path+filename+'.csv')
        return invalid_rows
    def col_ref_cover(data_raw,ref_dict):
        '''check the coverage of the reference values and extra values in the input data
        Args:
            data_raw: input dataframe
            ref_dict: required values that desire to exist in the data
        Returns:
             not_covered:dict of the uncovered values given the keys in the ref_dict
             extra:dict of the extra values given the keys in the ref_dict
        '''
        not_covered={}
        extra={}
        for key in ref_dict.keys():
            if key not in (data_raw.columns):
                print('The feature %s not found in data'%key)
            not_covered.update({key:ref_dict[key]})
        for col in data_raw.columns:
            if col not in ref_dict.keys():
                extra.update({col:data_raw[col].dropna().unique().tolist()})
            else:
                raw=set(data_raw[col].dropna().unique().tolist())
                ref=set(ref_dict[col])
                extra.update({col:list(raw-ref)})
                not_covered.update({col:list(ref-raw)})
            if len(not_covered[col])==0:
                not_covered.pop(col)
            if len(extra[col])==0:
                extra.pop(col)
        return not_covered,extra
    def timeseries_isIntegral(data_raw,start_time,finish_time):
        '''Checks whether a timeseris data integrally cover the period between start_time and finish_time

        Args:
            data_raw: Input timeseries with the index being timestamp
            start_time: The earliest timestamp required
            finish_time: The latest timestamp required

        Returns:
            isInteg: Bool,True if integrity test passed
            firt_time: The earliest timestamp of the data
            last_time: The latest timestamp of the data
        '''
        data=copy.deepcopy(data_raw)
        index=data.index.sort_values(ascending=True)
        first_time=index[0]
        last_time=index[-1]
        try:
            if first_time<=start_time and  last_time>=finish_time:
                isInteg=True
            else:
                isInteg=False
        except TypeError:
            print('The index of the input is not timestamp!')
            isInteg=None
        return isInteg,first_time,last_time

    def timeseries_isContinuous(data_raw,start_time,finish_time,frequency='D',
                                period=datetime.timedelta(days=1),missing_threshould=1,violation_threshould=1):
        '''Checks whether a timeseris data continuously cover the period between start_time and finish_time

        Args:
            data_raw: Input timeseries with the index being timestamp
            start_time: The earliest timestamp required
            start_time: The latest timestamp required
            frequency: The maximum interval between two records required
            period: Period to check the missing data
            missing_threshould: Times allowed for an continuous missing of the data during 1 period
            violation_threshould: Amount allowed for uncontinuous periods

        Returns:
            isContin: Bool,True if integrity test passed
            missing_index:missing timestamps
        '''
        data=copy.deepcopy(data_raw)
        index=data.index.sort_values(ascending=True)
        dummy=pd.date_range(start=start_time,end=finish_time,freq=frequency)
        missing_num=0
        violation_num=0
        pointer=dummy[0]
        violate_flag=False
        isContin=True

        for t in dummy:
            if t >= pointer+period:
                pointer+=period
                missing_num=0
                violate_flag=False
            if violate_flag:
                continue
            if not(t in index):
                missing_num+=1
            if missing_num>=missing_threshould:
                violation_num+=1
                missing_num=0
                violate_flag=True
            if violation_num>=violation_threshould:
                isContin=False
        missing_index=dummy[~dummy.isin(index)]
        return isContin,missing_index