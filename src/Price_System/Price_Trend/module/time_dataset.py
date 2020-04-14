'''
        File: time_dataset.py
        Date: 2020/3/27
        Author： herozen
        Version: v1.0
        Info：
            1. Tools to choose scope and generate dataset
            2. Function be import:
                    time_scope: Get time scope under time delta
                    time_dataset: Get dataset with scope choice
                
        Note:
                

'''

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



def time_dataset(data_raw, target, choice_delta, choice_num, time_index, date_raw):
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