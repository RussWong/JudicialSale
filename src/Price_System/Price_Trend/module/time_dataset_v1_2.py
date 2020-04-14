'''
        File: time_dataset_v1_2.py
        Date: 2020/4/3
        Author： herozen
        Version: v1.2
        Info：
            1. Tools to choose scope and generate dataset
            2. Function be import:
                    time_group_scope: Get time scope in different group and time delta
                    time_group_dataset：Get dataset in different group and time delta
            3. Other fuction：
                    time_scope: Get time scope under time delta.
                    time_dataset_scope: Get dataset with scope choice.
                    time_dataset_monyear: Get dataset with year/month scope.
                    time_dataset: Get dataset with feature generation.
            3. Function Tree
                1) time_group_scope： 
                    time_scope, time_monyear
                2) time_group_dataset:
                    time_scope -> time_dataset_scope -> time_dataset
                    time_monyear -> time_dataset 
                    

        Note:
            1. In time_dataset_monyear():
                     index of dataset is string.
               In time_dataset_scope(), index is datetime


        Version:
            v1.2 
                在v1.1基础上，
                修改time_monyear，加入模式选择，控制返回区间或数据
                新增time_group_scope: 观察不同粒度和时间间隔下的连续时间范围
                修改time_scope，加入check功能，判断是否最近时间起始、数据长度
                
            v1.1
                在v1.0基础上，将原time_dataset改名为time_dataset_scope
                新增time_dataset：将_dataset_scope得到的分组数据集，构建特征，生成可用数据集。
                新增time_monyear: 按年或月方式调整时间范围。
'''

import os
import numpy as np
import pandas as pd
from itertools import groupby

DATA_LEN = 20
TIMEDELTA = [1, 7, 30]
TIME_LEN = 100


def time_group_scope(data_raw, time_name, group_raw, timedelta):
    '''Get time scope in different group and time delta
    Args:
        data_raw: dataframe, whole data
        time_name: string, name of time, like 'Transaction_Time'
        group: dict, group choice, InnerType=list
        timedelta: list, time interval, InnerType=int, unit=days
    Return:
        time_index: dict, time scope in different group and time delta
    Note:
        1. It can be used to observe and help to make missing strategy.
            用于观察粒度和时间间隔下的数据，可观察到不连续情况，可考虑缺失填补。返回的时间范围更广
        2. 调用
            time_index = time_group_scope(data_raw=data, group_raw=group, time_name=time_name, timedelta=[7,15])
    '''

    data = data_raw.copy()
    group = group_raw.copy()

    time_index = {}
    _, time_index['All'] = time_scope(data[time_name], timedelta=timedelta)
    time_index['All'].update(time_dataset_monyear(data, time_name))

    try:
        for feature in group:
            if group[feature][0] == 'all':
                group[feature] = data[feature].unique()

            time_index[feature] = {}
            for value in group[feature]:
                data_tmp = data[data[feature] == value].copy()
                _, time_index[feature][value] = time_scope(data_tmp[time_name], timedelta=timedelta)
                time_index[feature][value].update(time_dataset_monyear(data_tmp, time_name))
    except:
        print('Wrong with group')

    return time_index






def time_group_dataset(data_raw, time_name, target_name, group_raw,
                         timedelta, data_path_raw, data_len = DATA_LEN):
    '''Get dataset in different group and time delta
    Args:
        data_raw: dataframe, whole data
        time_name: string, name of time, like 'Transaction_Time'
        target_name: string, name of target, like 'Final_Price'
        group: dict, group choice
        timedelta: list, time interval, InnerType=int, unit=days
        data_path_raw: string, path to store data, like './Dataset/'
        data_len: int, demand of data length

    Return:
        time_index: dict, time scope in different group and time delta
    Note:
        1. 先整体再粒度特征； 先time_scope再time_monyear
        2. 进行了check，只保留最近时间开始，且满足数据长度约束的数据
        3. 调用：
                dataset = time_group_dataset(data_raw=data, time_name=time_name,
                                 target_name=target_name, group_raw=group,
                                  timedelta=timedelta, data_path_raw=data_path_raw,
                                   data_len = DATA_LEN)           
    '''
    data = data_raw.copy()
    group = group_raw.copy()

    if not os.path.exists(data_path_raw):
        os.makedirs(data_path_raw)

    dataset = pd.DataFrame(columns=['Group', 'Timedelta' , 'Scope'])
    index = 0
    data_path = data_path_raw + 'dataset_' + str(index) + '.csv'

    # whole data
    # time scope
    date, time_index = time_scope(data[time_name], timedelta=timedelta,
                                 is_check=True, data_len=data_len)
    for delta in time_index:
        if time_index[delta]:
            data_scope = time_dataset_scope(data_raw=data, time_name=time_name, 
                                    time_scope=time_index[delta], timedelta=delta, date_raw=date)
            data_store = time_dataset(data_scope, target_name=target_name,
                                            time_name=time_name, choice=4)
            data_store.to_csv(data_path)

            dataset = dataset.append([{'Group':'All','Timedelta': delta,
                                'Scope':time_index[delta]}],ignore_index=True)
            index += 1
            data_path = data_path_raw + 'dataset_' + str(index) + '.csv'

    # time month and year
    data_year, data_month, time_index = time_dataset_monyear(data, time_name, mode=2)
    month = time_index['month'][0][0]
    year = time_index['year'][0][0]
    dataset = dataset.append([{'Group':'All','Timedelta': 'month',
                                'Scope':time_index['month']}],ignore_index=True)
    data_store = time_dataset(data_month, target_name=target_name,
                                    time_name=time_name, choice=4)
    data_store.to_csv(data_path)
    index += 1
    data_path = data_path_raw + 'dataset_' + str(index) + '.csv'

    dataset = dataset.append([{'Group':'All','Timedelta': 'year',
                                'Scope':time_index['year']}],ignore_index=True)
    data_store = time_dataset(data_year, target_name=target_name,
                                    time_name=time_name, choice=4)
    data_store.to_csv(data_path)
    index += 1
    data_path = data_path_raw + 'dataset_' + str(index) + '.csv'


    # data in group
    try:
        for feature in group:
            if group[feature][0] == 'all':
                group[feature] = data[feature].unique()

            time_index[feature] = {}
            for value in group[feature]:
                data_tmp = data[data[feature] == value].copy()
                group_name = feature + '==' + value
                
                # time scope
                date, time_index = time_scope(data_tmp[time_name], timedelta=timedelta,
                                                is_check=True, data_len=data_len)
                for delta in time_index:
                    if time_index[delta]:
                        data_scope = time_dataset_scope(data_raw=data_tmp, time_name=time_name, 
                                            time_scope=time_index[delta], timedelta=delta,
                                             date_raw=date)
                        data_store = time_dataset(data_scope, target_name=target_name,
                                                         time_name=time_name, choice=4)
                        data_store.to_csv(data_path)
                        dataset = dataset.append([{'Group': group_name ,'Timedelta': delta,
                                            'Scope':time_index[delta]}],ignore_index=True)
                        index += 1
                        data_path = data_path_raw + 'dataset_' + str(index) + '.csv'

                # time month and year
                data_year, data_month, time_index = time_dataset_monyear(data_tmp, time_name, mode=2)
                # check time
                if time_index['month'][0][0] == month:
                    dataset = dataset.append([{'Group': group_name,'Timedelta': 'month',
                                                'Scope':time_index['month']}],ignore_index=True)
                    data_store = time_dataset(data_month, target_name=target_name,
                                                    time_name=time_name, choice=4)
                    data_store.to_csv(data_path)
                    index += 1
                    data_path = data_path_raw + 'dataset_' + str(index) + '.csv'

                if time_index['year'][0][0] == year:
                    dataset = dataset.append([{'Group':group_name,'Timedelta': 'year',
                                                'Scope':time_index['year']}],ignore_index=True)
                    data_store = time_dataset(data_year, target_name=target_name,
                                                    time_name=time_name, choice=4)
                    data_store.to_csv(data_path)
                    index += 1
                    data_path = data_path_raw + 'dataset_' + str(index) + '.csv'
    except:
        print('Wrong with group')

    return dataset




def time_scope(data_raw, timedelta=TIMEDELTA, time_len=TIME_LEN, is_check=False, data_len=DATA_LEN ):
    '''Find time scope under time delta
    
    Args:
        data_raw: Series, data time, InnerType = string
        timedelta: list, default=[1, 7, 30], InnerType=int, unit=days
        time_len: int, to decide length of time series, unit=days
        is_check: bool, whether check time start and data length
        data_len: int, demand of data length
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
        index_tmp = time_scope_calculate(time_countinue, 100, date['Date'])
        if is_check:
            start = pd.to_datetime(index_tmp[0][0])
            end = pd.to_datetime(index_tmp[0][1])
            if start == date['Date'][0] and (start-end).days/i > data_len:
                time_index[i] = index_tmp[0]
        else:
            time_index[i] = index_tmp

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




def time_dataset_scope(data_raw, time_name, time_scope, timedelta, date_raw):
    '''Get dataset with scope and interval choice

    Args:
        data_raw: dataframe, whole data. 
        time_name: string, column name of time data
        time_scope: list, time scope
        timedelta: int, time delta
        date: dataframe, return by time_scope
    Return:
        data: dataframe, dataset used by next step.
        
    '''
    data = data_raw.copy()
    date = date_raw.copy()

    start = pd.to_datetime(time_scope[0])
    end = pd.to_datetime(time_scope[1])

    date['Code'] = (date['Date'][0] - date['Date']).\
                apply(lambda x: x.days).floordiv(timedelta)
    date.set_index('Date', inplace=True)

    # From original datetime to num
    smooth_day = date.loc[start:end, 'Code']
    # From num to interval datetime
    smooth_range = pd.Series(pd.date_range(end=start, freq=str(timedelta)+'D', 
                                periods=smooth_day.max() + 1))
    smooth_range.sort_values(ascending=False, inplace=True)
    smooth_range.reset_index(drop=True, inplace=True)
    smooth_day = smooth_day.map(smooth_range)

    data[time_name] = pd.to_datetime(data[time_name])     # change to datetime
    data[time_name] = data[time_name].map(smooth_day)     # code

    # delete null
    data = data[data[time_name].notnull()]
    data.reset_index(drop=True, inplace=True)

    # set time index
    data.set_index(time_name, inplace=True)

    return data




def time_dataset_monyear(data_raw, time_name, mode = 1):
    ''' Get dataset with year/month scope. And missing
    
    Args:
        data_raw: dataframe, whole data
        time_name: string, column name of time
        mode: int, control return
                1: return time scope
                2: return data
    Return:
        data_year: dataframe, index is date in year
        data_month: dataframe, index is date in month
        missing: dict, missing value


    Note:
        1. Use different way to time_dataset_scope. 
           Because year and month is hard to calculate in time type.

    '''
    
    data = data_raw[data_raw[time_name].notnull()].copy()  # default without NaN
    # transform type
    time_month = pd.to_datetime(data[time_name]).dt.strftime('%Y-%m') 
    time_year = pd.to_datetime(data[time_name]).dt.strftime('%Y')
    # change index
    data_year = data.copy()
    data_month = data.copy()
    data_year[time_name] = time_year
    data_month[time_name] = time_month
    data_year.set_index(time_name, inplace=True)
    data_month.set_index(time_name, inplace=True)

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

    time_index = {}
    tmp = data_month.index.unique()
    time_index['month'] = [[tmp.max(), tmp.min()]]
    tmp = data_year.index.unique()
    time_index['year'] = [[tmp.max(), tmp.min()]]

    if mode == 1:
        return time_index
    elif mode == 2:
        return data_year, data_month, time_index




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



