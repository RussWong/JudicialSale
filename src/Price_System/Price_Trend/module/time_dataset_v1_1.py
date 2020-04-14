'''
        File: time_dataset.py
        Date: 2020/3/30
        Author： herozen
        Version: v1.1
        Info：
            1. Tools to choose scope and generate dataset
            2. Function be import:
                    time_scope: Get time scope under time delta.
                    time_dataset_scope: Get dataset with scope choice.
                    time_dataset_monyear: Get dataset with year/month scope.
                    time_dataset: Get dataset with feature generation.
            3. Function Tree
                1) time_scope -> time_dataset_scope -> time_dataset
                2) time_monyear -> time_dataset 

        Note:
            1. In time_dataset_monyear():
                     index of dataset is string.

               In time_dataset_scope(), index is datetime


        Version:
            v1.1
                在v1.0基础上，将原time_dataset改名为time_dataset_scope
                新增time_dataset：将_dataset_scope得到的分组数据集，构建特征，生成可用数据集。
                新增time_monyear: 按年或月方式调整时间范围。
                

'''

import numpy as np
import pandas as pd
from itertools import groupby



def time_count(data_raw):
    '''Count time distribution
       
    Args:
        data_raw: Series, data time 
    Return:
        year: series, year distribution
        month: seires, year-month distribution
    '''

    data = pd.to_datetime(data_raw.copy())
    year = data.dt.strftime('%Y').value_counts()
    month = data.dt.strftime('%Y-%m').value_counts()
    return year, month




TIMEDELTA = [1, 7, 30]
TIME_LEN = 100

def time_scope(data_raw, timedelta=TIMEDELTA, time_len=TIME_LEN):
    '''Find time scope under time delta
    
    Args:
        data_raw: Series, data time, InnerType = string
        timedelta: list, default=[1, 7, 30], InnerType=int, unit=days
        time_len: int, to decide length of time series, unit=days
    Reuturn:
        date: dataframe, unique time and time interval
        time_index: dict, time scope in different time delta

    '''

    data = data_raw.copy()
    date = pd.Series(data.dropna().unique())    # default without NaN
    date.sort_values(ascending=False, inplace=True)
    date.reset_index(drop=True, inplace=True)
    date = pd.to_datetime(date)
    date = pd.DataFrame(date,columns=['Date'])
    date['Time_Delta'] = date['Date'] - \
                         date.loc[1:,'Date'].copy().reset_index(drop=True)

    time_index = {}
    for i in timedelta:
        delta = pd.Timedelta(value=i, unit='D')
        time_countinue = (date['Time_Delta'] <= delta)
        time_index[i] = time_scope_calculate(time_countinue, time_len, date['Date'])

    return date, time_index




def time_scope_calculate(data_raw, time_len, date):
    '''

    Args:
        data_raw: Series, InnerType = bool 
        date: Seires, date
    '''
    scope = []
    lst = list(data_raw[data_raw].index)
    fun = lambda x: x[1]-x[0]
    
    for k, g in groupby(enumerate(lst), fun):
        l1 = [j for i, j in g]   
        if len(l1) + 1 >= time_len:
            scope.append([date[min(l1)].strftime('%Y-%m-%d'),\
                         date[max(l1) + 1].strftime('%Y-%m-%d')])    

    return scope




# # 观察时间区间的数据分布
# import matplotlib.pyplot as plt
# b = data.value_counts().sort_index()
# plt.figure(figsize=tuple(2 * np.array([6.4, 9.6])))
# plt.yticks(fontsize=5)
# plt.barh(list(b[time_index['1D'][0][1]:time_index['1D'][0][0]].index), b[time_index['1D'][0][1]:time_index['1D'][0][0]])



def time_dataset_scope(data_raw, target, choice_delta, choice_num, time_index, date_raw):
    '''Get dataset with scope and interval choice

    Args:
        data_raw: dataframe, whole data. 
        target: string, column name of time data
        choice_delta: int, time delta
        choice_num: int, which scope in time_index
        time_index: dict, return by time_scope
        date: dataframe, return by time_scope
    Return:
        data: dataframe, dataset used by next step.
        
    '''
    data = data_raw.copy()
    date = date_raw.copy()

    start = pd.to_datetime(time_index[choice_delta][choice_num-1][0])
    end = pd.to_datetime(time_index[choice_delta][choice_num-1][1])

    date['Code'] = (date['Date'][0] - date['Date']).\
                apply(lambda x: x.days).floordiv(choice_delta)
    date.set_index('Date', inplace=True)

    # From original datetime to num
    smooth_day = date.loc[start:end, 'Code']
    # From num to interval datetime
    smooth_range = pd.Series(pd.date_range(end=start, freq=str(choice_delta)+'D', 
                                periods=smooth_day.max() + 1))
    smooth_range.sort_values(ascending=False, inplace=True)
    smooth_range.reset_index(drop=True, inplace=True)
    smooth_day = smooth_day.map(smooth_range)

    data[target] = pd.to_datetime(data[target])     # change to datetime
    data[target] = data[target].map(smooth_day)     # code

    # delete null
    data = data[data[target].notnull()]
    data.reset_index(drop=True, inplace=True)

    # set time index
    data.set_index(target, inplace=True)

    return data



def time_dataset_monyear(data_raw, time_column):
    ''' Get dataset with year/month scope. And missing
    
    Args:
        data_raw: dataframe, whole data
        time_column: string, column name of time

    Return:
        data_year: dataframe, index is date in year
        data_month: dataframe, index is date in month
        missing: dict, missing value


    Note:
        1. Use different way to time_dataset_scope. 
           Because year and month is hard to calculate in time type.

    '''
    
    data = data_raw[data_raw[time_column].notnull()].copy()  # default without NaN
    # transform type
    time_month = pd.to_datetime(data[time_column]).dt.strftime('%Y-%m') 
    time_year = pd.to_datetime(data[time_column]).dt.strftime('%Y')
    # change index
    data_year = data.copy()
    data_month = data.copy()
    data_year[time_column] = time_year
    data_month[time_column] = time_month
    data_year.set_index(time_column, inplace=True)
    data_month.set_index(time_column, inplace=True)

    # count missing time
    # parameters
    ## time_year_counts = time_year.value_counts()     # can be used for observe.
    ## time_month_counts = time_month.value_counts()
    time_year_index = time_year.unique()
    time_month_index = time_month.unique()
    # reference year and month
    year = pd.Series(pd.date_range(start = time_year_index.min(),
                     end = time_year_index.max(), freq='1Y')).dt.strftime('%Y').tolist()
    month = pd.Series(pd.date_range(start = time_month_index.min(),
                     end = time_month_index.max(), freq='1M')).dt.strftime('%Y-%m').tolist()
    # missing
    missing = {}
    missing['year'] = [x for x in year if x not in time_year_index]
    missing['month'] = [x for x in month if x not in time_month_index]

    # adjust dataset about continuity
    if missing['month']:
        time_month_index = time_month_index[time_month_index > missing['month'][-1]]
        data_month = data_month.loc[time_month_index,:]
    if missing['year']:
        time_year_index = time_year_index[time_year_index > missing['year'][-1]]
        data_year = data_year.loc[time_year_index,:]

    return data_year, data_month, missing


def time_dataset(data_raw, target_name, time_name, choice = 4):
    ''' Get dataset with feature generation.

    Args:
        data_raw: dataframe, whole data
        target_name: string, name of target, like 'Final_Price'
        time_name: string, name of time, like 'Transaction_Time'
        choice: int, choice of feature generation
                1: median of target.
                2: median of target, counts of data
                3: median of target, counts of data, mean of target
                4: median, mean, sum, max, min, std of target. And counts of data
    Return:
        data: dataframe, the regenerated data with features
    '''

    data_input = data_raw.copy()
    
    if choice == 1:
        data = data_input.groupby(time_name)[target_name].agg([np.median])
    elif choice == 2:
        data = data_input.groupby(time_name)[target_name].agg([np.median])
        count = pd.Series(data_input.index).value_counts()
        count = count.rename('counts')
        data = pd.concat([data,count],axis=1, sort=False)
    elif choice == 3:
        data = data_input.groupby(time_name)[target_name].\
                            agg([np.median, 'mean'])
        count = pd.Series(data_input.index).value_counts()
        count = count.rename('counts')
        data = pd.concat([data,count],axis=1, sort=False)
    elif choice == 4:
        count = pd.Series(data_input.index).value_counts()
        count = count.rename('counts')
        data = data_input.groupby(time_name)[target_name].\
                            agg([np.median, 'mean', 'sum', 'max','min',np.std])
        data = pd.concat([data,count],axis=1, sort=False)

    # missing analysis
    missing = data.isnull().sum()
    missing = missing[missing > 0].index.tolist()
    if missing:
        print('Note: missing value in {}'.format(missing))
    return data

