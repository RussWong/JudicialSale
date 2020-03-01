#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
# pd.set_option('display.max_columns',None)



def anomaly_analysis(data_raw, ylabel, data_num=10, 
                     pre_clean = False, path='./', 
                     plot_score=True, plot_anomal=True):
    '''
        Info:
            Top function for anomaly analysis
        IO:
            Input:
                    data_raw: data to detect
                    ylabel: label of target
                    data_num: number of Top anomal data to return
                    pre_clean: if True, data 
                    path: path to store picture
                    plot_score:  whether to plot anomal score distribution
                    plot_anomal: whether to  plot anomal data
            Output:
                    Anomaly_Score.png

    '''
    print('======== Anomaly anlysis start ========')

    data = data_raw.copy()

    if pre_clean == False:
        anomal_data_pre(data)
        
    # select numerical column 
    data_types = data.dtypes
    col_num = [data_types.index[i] for i in range(len(data_types)) if data_types[i] in ['int', 'float']]

    anomaly_detect(data, col_num)

    if plot_score:
        anomal_plot_score(data['score'].values, path)
    if plot_anomal:
        anomal_plot_anomal(data, col_num, ylabel, path)

    # TOP Anomal data store
    data.sort_values(by='score').head(data_num).\
         drop('Outlier', axis=1).\
         to_csv(path + 'Top_Anomal_Data.csv', index=False)

    
    print('======== Anomaly anlysis end ========')



def anomal_data_pre(data):
    '''
        Info：
            preprocessing for data. Including duplicate and missing
        IO:
            Input:
                    data_raw: data to detect

    '''
    print('==== Remove duplicate ====')

    print('Shape before：' + str(data.shape))
    data.drop_duplicates(keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('Shape after：' + str(data.shape))

    print('==== Remove missing ====')
    # Remove feature if miss percent > 10%
    print('Shape before：' + str(data.shape))
    missing = data.isnull().sum() / data.shape[0]
    col_drop = [missing.index[i] for i in range(len(missing)) if missing[i] > 0.1]
    data.drop(col_drop, axis=1, inplace=True)
    print('Shape inter：' + str(data.shape))
    print('Column drop：' + str(col_drop))
    # Remove other missing data
    data.dropna(inplace=True)
    print('Shape after：' + str(data.shape))


    

def anomaly_detect(data, cols):
    '''
        Info：
            model to detect anomaly
        IO:
            Input:
                    data: data to detect
                    cols: columns for numerical data
            Returnt:
                    res: index of anomal data
                    score: anomal score of data
    '''
    print('==== Train model ====')
    res = []

    for i in range(100):
        clf = IsolationForest(behaviour='new', contamination='auto')
        pred = clf.fit_predict(data[cols])
        data['Outlier'] = pred

        if i == 0: 
            res = list(data[data['Outlier'] == -1].index)
            continue

        tmp = list((set(res) & set(list(data[data['Outlier'] == -1].index))))

        if len(tmp) == len(res) or i == 99:
            print('Detection Finished. Iteration num:',i)
            data['score'] = clf.decision_function(data[cols].values)
            break
        else:
            res = tmp



            
def anomal_plot_score(score, path):
    '''
        Info:
            Plot anomal score distribution
    '''

    print('==== Plot anomaly score ====')

    plt.figure()
    plt.hist(score)
    plt.title('Anomaly Score Distribution')
    plt.xlabel('scores')
    plt.ylabel('count')
    plt.savefig(path+'Anomal_Score.png')






def anomal_plot_anomal(data_raw, cols, ylabel, path):
    '''
        Info:
            Plot anomal data (numerical)
    '''
    print('==== Plot anomal data ====')

    data = data_raw[cols].copy()
    num = len(data.columns)
    plt.figure(figsize=(6.4, num * 4.8))

    for i in range(num):
        plt.subplot(num, 1, i + 1, title='Figure ' + str(i + 1))
        plt.xlabel(list(data.columns)[i])
        plt.ylabel(ylabel)
        plt.scatter(data.iloc[:,i], data.loc[:, ylabel], s=0.4, c=data_raw['Outlier'])
        
    plt.subplots_adjust(top=0.965)
    plt.suptitle('Anomal_Data')
    plt.savefig(path + 'Anomal_Data.png')

