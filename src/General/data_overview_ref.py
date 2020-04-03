'''
        Date: 2020/3/13
        Author： herozen
        Version: v1.0
        Info：
                


        Note:
                1. 注意.value_count() 不会统计NaN
                2. 待解决问题：
                    1）dataframe中混杂数据类型

'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


FILE_PATH = './myplot/'
FIGURE_SIZE = 1.4
FONT_SIZE = 1

def count_missing(raw_data, is_plot=True, is_save=False,
                    figure_size=FIGURE_SIZE, font_size=FONT_SIZE,
                    file_path=FILE_PATH, file_name='count_missing'):
    '''
        Info:
            count missing for data.
        Output:
            missing_count: dataframe
            figure: png, stored in FILE_PATH + file_name + '.png'
        Note:
            the output can be used like:
                missing_count[missing_count.iloc[:,1] > threshold]
    '''

    data = raw_data.copy()
    
    # count missing
    missing = data.isnull().sum() 
    missing.sort_values(inplace=True, ascending=False)
    percent = round(missing / data.shape[0], 4) * 100
    missing_count = pd.concat([missing.to_frame(name='缺失数'), \
                            percent.to_frame(name='缺失百分比(%)')], axis=1)

    # plot figure
    if is_plot:
        data_plot = missing_count.iloc[::-1].copy()
        plt.figure(figsize=tuple(figure_size * np.array([6.4, 4.8])))
        barh = plt.barh(list(data_plot.index), data_plot.iloc[:,1])
        for rect in barh:
            wid = rect.get_width()
            plt.text(wid, rect.get_y() + rect.get_height()/2, 
                        '%.2f'%(wid), va='center', fontsize=font_size * 10)
        
        plt.xlim([0, data_plot.max()[1] * 1.08])
        plt.yticks(ticks=list(data_plot.index), fontsize=font_size * 10)
        plt.xticks(fontsize=font_size * 15)
        plt.xlabel('缺失率(%)', fontsize=font_size * 15)
        plt.title('缺失率情况统计', fontsize=font_size * 15)
        plt.tight_layout()
        if is_save:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            plt.savefig(file_path + file_name + '.png')

    return missing_count




def count_unique(raw_data, is_plot=True, is_save=False,
                    figure_size=FIGURE_SIZE, font_size=FONT_SIZE,
                    file_path=FILE_PATH, file_name='count_unique'):
    '''
        Info:
            count unique value for each feature.
        Output:
            unique_count: dataframe
            figure: png, stored in FILE_PATH + file_name + '.png'
        Note:
            the output can be used like:
                list(unique_count[unique_count.iloc[:,0] < threshold].index)
                
    '''

    data = raw_data.copy()
    
    # count unique value
    unique = pd.Series()
    for col in data.columns:
        unique[col] = len(data[col].unique())
    unique_count = pd.DataFrame(unique, columns=['独立值个数'])

    if is_plot:
        data_plot = unique_count.iloc[::-1].copy()
        plt.figure(figsize=tuple(figure_size * np.array([6.4, 4.8])))
        barh = plt.barh(list(data_plot.index), data_plot.iloc[:,0])
        for rect in barh:
            wid = rect.get_width()
            plt.text(wid, rect.get_y() + rect.get_height()/2, '%d'%(wid), va='center', fontsize=font_size * 10)

        plt.xlim([0, data_plot.max()[0] * 1.08])
        plt.yticks(ticks=list(data_plot.index), fontsize=font_size * 10)
        plt.xticks(fontsize=font_size * 15)
        plt.xlabel('独立值个数', fontsize=font_size * 15)
        plt.title('特征独立值情况统计', fontsize=font_size * 15)
        plt.tight_layout()
        if is_save:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            plt.savefig(file_path + file_name + '.png')

    return unique_count




def plot_distribution(raw_data, threshold=10, is_save=False, 
                    figure_size=FIGURE_SIZE, font_size=FONT_SIZE,
                    file_path=FILE_PATH, file_name='data distribution'):
    '''
        Info:
            Plot data distribution, with counting unique
        Input:
            raw_data: dataframe
            threshold: int, to pick out categorical data from numerical data
        Output:
            figure: png, stored in FILE_PATH + file_name + '.png'
        Note:   
            Assume data type is correct, expecially for timestamp
    '''

    data = raw_data.copy()

    # Distiguish data type
    data_dtypes = data.dtypes
    col_num = [data_dtypes.index[i] for i in range(len(data_dtypes)) if data_dtypes[i] in ['int', 'float']]
    col_str = [data_dtypes.index[i] for i in range(len(data_dtypes)) if data_dtypes[i] in ['O']]
    col_other = list(set(list(data.columns)).difference(set(col_num + col_str)))
    data_types = {}
    data_types['num'] = col_num
    data_types['str'] = col_str
    data_types['other'] = col_other

    # choose part numerical to categorical
    unique = pd.Series()
    for col in data.columns:
        unique[col] = len(data[col].unique())
    unique_count = pd.DataFrame(unique, columns=['Unique'])

    col_tmp = list(unique_count[unique_count.iloc[:,0] <= threshold].index)
    col_num = list(set(list(col_num)).difference(set(col_tmp)))
    col_str = list(set(col_str).union(set(col_tmp)))


    # plot figure
    fig_num = len(col_num + col_str) 
    plt.figure(figsize=tuple(figure_size * np.array([6.4, 4.8 * fig_num])))
    
    # plot annotation
    # plt.subplot(fig_num, 1, 1)
    # text = '判定门限：' + str(threshold) + '\n' + \
    #         '数值型数据量' + str(len(col_num)) + '\n' + \
    #         '字符型数据量' + str(len(col_str)) + '\n' + \
    #         '其他类型特征' + str(len(col_other))
    # plt.text(x=0.2, y=0.8, s=text, fontsize=font_size * 15 )
    # plt.xticks([])
    # plt.yticks([])
    # 可修改：划分网格的方式显示

    # plot numerical
    if col_num:
        for i in range(len(col_num)):
            miss_num = data[col_num[i]].isnull().sum()
            data_tmp = data[col_num[i]].dropna().copy()
            data_num = data_tmp.shape[0]
            text = '图中数据量：' + str(data_num) + '\n' +\
                   '缺失数据量：' + str(miss_num)
            plt.subplot(fig_num, 1, i + 1)
            plt.hist(data_tmp)
            plt.xticks(fontsize=font_size * 15)
            plt.yticks(fontsize=font_size * 15)
            plt.xlabel('特征取值',fontsize=font_size * 20)
            plt.ylabel('频数',fontsize=font_size * 20)
            plt.title('特征('+ str(col_num[i]) +')数据分布', fontsize=font_size * 20)
            bottom_y, top_y = plt.ylim()
            bottom_x, top_x = plt.xlim()
            plt.ylim(bottom_y, top_y * 1.15)
            axis_y = top_y
            axis_x = (top_x - bottom_x) * 0.01 + top_x
            plt.text(x=axis_x, y=axis_y, s=text, fontsize=font_size * 12)
        plt.tight_layout()
    if col_str:
        for i in range(len(col_str)):
            
            miss_num = data[col_str[i]].isnull().sum()
            data_tmp = data[col_str[i]].fillna('缺失').copy()
            data_tmp = data_tmp.apply(lambda x : str(x))        # change type
            data_num = data_tmp.shape[0]
            text = '图中数据量：' + str(data_num) + '\n' +\
                   '缺失数据量：' + str(miss_num)
            data_value = data_tmp.value_counts()
            data_value.sort_values(ascending=False)

            if len(data_value) > threshold:
                data_sum = data_value[threshold-1:].sum()
                data_value = data_value[0:threshold-1]
                data_value['其他数据'] = data_sum
            xlabel = list(data_value.index)

            plt.subplot(fig_num, 1, i + len(col_num) + 1)
            plt.bar(xlabel, data_value.values)
            plt.xticks(rotation=90, fontsize=font_size * 15)
            plt.yticks(fontsize=font_size * 15) 
            plt.xlabel('特征取值',fontsize=font_size * 20)
            plt.ylabel('频数',fontsize=font_size * 20)
            plt.title('特征('+ str(col_str[i]) +')数据分布', fontsize=font_size * 20)
            bottom_y, top_y = plt.ylim()
            bottom_x, top_x = plt.xlim()
            plt.ylim(bottom_y, top_y * 1.15)
            axis_y = top_y
            axis_x = (top_x - bottom_x) * 0.01 + top_x
            plt.text(x=axis_x, y=axis_y, s=text, fontsize=font_size * 12)
        plt.tight_layout()

        if is_save:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            plt.savefig(file_path + file_name + '.png')
            
    return data_types
