import luigi
import pandas as pd 
import numpy as np 
#import type_of_dataset
import sys
import joblib
import shap
import matplotlib.pyplot as plt

from analysis import globalSurrogate,shap_compute,residuals_plot
from handpreprocessing import handpreprocessing
from fucset import data_duplicate, data_normalization, data_anomaly, data_encoding, data_encoding_2
from missing import missing_value_processing
from train import Model_train
from data_overview import plot_missing,plot_col_unique,plot_numhist,plot_strbar,plot_correlation

from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor

# Task 0 人工预处理
# TODO 读取数据，手动特征工程

class Hand(luigi.Task):
    # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名

    # 
    def requires(self):
        return None

    def output(self):
        return {'data' : luigi.LocalTarget('Dataset/{0}/{1}/{1}_hand.csv'.format(self.type_of_dataset, self.name_of_dataset)),
        'cols' : luigi.LocalTarget('Dataset/{0}/{1}/{1}_cols.txt'.format(self.type_of_dataset, self.name_of_dataset))}

    def run(self):
        raw_data = pd.read_csv('Dataset/{0}/{1}/{1}.csv'.format(self.type_of_dataset, self.name_of_dataset))

        if self.is_need == 1:
            
            hand = handpreprocessing(data=raw_data, type_of_dataset=self.type_of_dataset, cols_num=self.cols_num)
            data_pre, cols = hand.run()
            with self.output()['data'].open('w') as outfile:
                data_pre.to_csv(outfile, encoding='utf-8', index=False)
            with self.output()['cols'].open('w') as outfile:
                for key, value in cols.items():
                    outfile.write(str(key) + '-' + str(value) + '\n')
                    
        else:
            cols = {}
            cols['num'] = self.cols_num
            cols['str'] = list(set(list(raw_data.columns)).difference(set(cols['num'])))
            with self.output()['data'].open('w') as outfile:
                raw_data.to_csv(outfile, encoding='utf-8', index=False)
            with self.output()['cols'].open('w') as outfile:
                for key, value in cols.items():
                    outfile.write(str(key) + '-' + str(value) + '\n')


# Task 1 数据预处理
# TODO
class Preprocessing(luigi.Task):
    # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名

    def requires(self):
        return Hand(is_need=self.is_need, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, cols_num=self.cols_num)
    
    def output(self):
        return {'data' : luigi.LocalTarget('Dataset/{0}/{1}/{1}_preprocessing.csv'.format(self.type_of_dataset, self.name_of_dataset)),
        'cols' : luigi.LocalTarget('Dataset/{0}/{1}/{1}_cols_pre.txt'.format(self.type_of_dataset, self.name_of_dataset))}
    
    def run(self):
        raw_data = pd.read_csv(self.input()['data'].path)
        cols = {}
        with open(self.input()['cols'].path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                key = line.split('-')[0]
                value = eval(line.split('-')[1])
                cols[key] = value
    
        # TODO 1.1 数据概述
        plot_missing(raw_data,'Dataset/{0}/{1}/dataoverview_picutures/'.format(self.type_of_dataset, self.name_of_dataset),'missing_rate')
        plot_col_unique(raw_data,'Dataset/{0}/{1}/dataoverview_picutures/'.format(self.type_of_dataset, self.name_of_dataset),'unique_count')
        plot_numhist(raw_data[cols['num']].dropna(axis=1),'Dataset/{0}/{1}/dataoverview_picutures/'.format(self.type_of_dataset, self.name_of_dataset),'num_varb_distribution')
        plot_strbar(raw_data[cols['str']].dropna(axis=1),'Dataset/{0}/{1}/dataoverview_picutures/'.format(self.type_of_dataset, self.name_of_dataset),'str_varb_distribution')
        plot_correlation(raw_data.dropna(axis=1),'Dataset/{0}/{1}/dataoverview_picutures/'.format(self.type_of_dataset, self.name_of_dataset),'corelation_heatmap')        
        # TDO 1.2 数据去重
        data_duplicate(raw_data)
        
        # TODO 1.3 数据标准化
        tmp = cols['num'].copy()
        tmp.remove(self.name_of_target)
        data_normalization(raw_data, tmp)
        
        # TODO 1.4 缺失值处理
        missing_pre = missing_value_processing(raw_data, cols)
        for key in cols.keys():
            for x in cols[key]:
                if x not in missing_pre.columns:
                    cols[key].remove(x)

        with self.output()['data'].open('w') as f:
            missing_pre.to_csv(f, index=False, encoding='utf-8')
        
        with self.output()['cols'].open('w') as outfile:
                for key, value in cols.items():
                    outfile.write(str(key) + '-' + str(value) + '\n')

# Task2 异常检测
# TODO 主要使用孤立森林
class Anomaly(luigi.Task):
    # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名


    def requires(self):
        return Preprocessing(is_need=self.is_need, type_of_dataset=self.type_of_dataset, 
        name_of_dataset=self.name_of_dataset, name_of_target=self.name_of_target, cols_num=self.cols_num)

    def output(self):
        return {'data' : luigi.LocalTarget('Dataset/{0}/{1}/{1}_anomaly.csv'.format(self.type_of_dataset, self.name_of_dataset))}
        

    def run(self):
        data = pd.read_csv(self.input()['data'].path)
        print(data.columns)
        cols = { }
        with self.input()['cols'].open('r') as f:
            for line in f.readlines():
                line = line.strip()
                key = line.split('-')[0]
                value = line.split('-')[1]
                cols[key] = eval(value)
        data = data_anomaly(data, cols, self.name_of_target, 
        'Dataset/{0}/{1}/anomaly_picture/'.format(self.type_of_dataset, self.name_of_dataset))
        with self.output()['data'].open('w') as f:
            data.to_csv(f, index=False, encoding='utf-8')

# Task 3 编码
# TODO
class Encoding(luigi.Task):
    # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名


    def requires(self):
        return {'anomaly' : Anomaly(is_need=self.is_need, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_target=self.name_of_target, cols_num=self.cols_num),
            'preprocessing' :Preprocessing(is_need=self.is_need, type_of_dataset=self.type_of_dataset, 
        name_of_dataset=self.name_of_dataset, name_of_target=self.name_of_target, cols_num=self.cols_num)
        }
    
    def output(self):
        return {'data' : luigi.LocalTarget('Dataset/{0}/{1}/{1}_encoding.csv'.format(self.type_of_dataset, self.name_of_dataset))}

    def run(self):
        data = pd.read_csv(self.input()['anomaly']['data'].path)
        cols = {}
        with self.input()['preprocessing']['cols'].open('r') as f:   # anomaly未改变cols
            for line in f.readlines():
                line = line.strip()
                key = line.split('-')[0]
                value = eval(line.split('-')[1])
                cols[key] = value
        print(data.columns, len(data.columns))
        print(cols, len(cols['str']) + len(cols['num']))

        if self.type_of_encoding == 'target':
            data = data_encoding_2(data, cols, self.name_of_target)
        elif self.type_of_encoding == 'one-hot':
            data = data_encoding(data, cols)
        
        with self.output()['data'].open('w') as f:
            data.to_csv(f, index=False, encoding='utf-8')

# Task 4 特征选择
# TODO
class FeatureSelection(luigi.Task):
    # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名

#    !  ------------------------------------------+++++++++++-----------------------------------
    num_of_feature = luigi.IntParameter(default=15)
    import warnings
    warnings.filterwarnings("ignore")
    
    def requires(self):
        return Encoding(is_need=self.is_need, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_target=self.name_of_target,
        type_of_encoding=self.type_of_encoding, cols_num=self.cols_num)
    
    def output(self):
        return {'data' : luigi.LocalTarget('Dataset/{0}/{1}/{1}_feature.csv'.format(self.type_of_dataset, self.name_of_dataset))}

    def run(self):
        data = pd.read_csv(self.input()['data'].path)

        # x除去价格列的数据，y价格列；x_feature特征list
        y = data[self.name_of_target].values
        x = data.drop([self.name_of_target], axis=1).values
        x_feature = list(data.columns)
        x_feature.remove(self.name_of_target)     
        
        # 方法一：单变量特征选择 f_regression 基于方差
        f_test, _ = f_regression(x, y)
        result_f_regression = list(zip([i for i in range(len(x_feature))], f_test))
        result_f_regression.sort(key = lambda x: x[1], reverse = True)
        fr_ranking = [0] * len(x_feature)
        rank = 1
        for i in result_f_regression:
            fr_ranking[i[0]] = rank
            rank += 1
       
        # 方法二：单变量特征选择 mutual_info_regression 基于互信息
        mutual_test = mutual_info_regression(x, y)
        result_mutual_info_regression = list(zip([i for i in range(len(x_feature))], mutual_test))
        result_mutual_info_regression.sort(key = lambda x: x[1], reverse = True)
        mu_ranking = [0] * len(x_feature)
        rank = 1
        for i in result_mutual_info_regression:
            mu_ranking[i[0]] = rank
            rank += 1      
        
        # RFE的稳定性很大程度上取决于迭代时，底层用的哪种模型。
        # 没有经过正则化的回归是不稳定的，比如普通的回归（LR），那么RFE就是不稳定的。
        # 假如采用的是Lasso/Ridge，正则化的回归是稳定的，那么RFE就是稳定的。
        
        # 方法三：基于RFE方法，评估器使用LR线性回归
        rfe0 = RFE(estimator=LogisticRegression(), n_features_to_select = 1)
        rfe0.fit(x, y) 
        
        # 方法四：基于RFE方法，评估器使用Ridge回归,CV表示交叉验证
        rcv = RidgeCV([1, 2, 5, 10, 20, 100])
        rfe1 = RFE(estimator=rcv, n_features_to_select = 1)
        rfe1.fit(x, y)
        
        # 方法五：基于RFE方法，评估器使用Lasso回归,CV表示交叉验证
        rfe2 = RFE(estimator=LassoCV(), n_features_to_select = 1)
        rfe2.fit(x, y)
        
        # 方法六：基于RFE方法，评估器使用随机森林RandomForestRegressor
        rfe3 = RFE(estimator=RandomForestRegressor(), n_features_to_select = 1)
        rfe3.fit(x, y)

        feature_select = pd.DataFrame([fr_ranking, mu_ranking, list(rfe0.ranking_), list(rfe1.ranking_), list(rfe2.ranking_), list(rfe3.ranking_)],index=['f_regression', 'mutual_info_regression', 'RFE_LR', 'RFE_RidgeCV', 'RFE_LassoCV', 'RFE_RandomForestRegressor'],columns=x_feature)
                
        l = []#删除特征
        for i in x_feature:
            tmp1 = feature_select[i].map(lambda x: x > self.num_of_feature ).sum()
            tmp2 = feature_select[i].sum()
            l.append([i ,tmp1 + tmp2 / 100])
        l.sort(key = lambda x: x[1])
        for i in l[self.num_of_feature:]:
            data.drop(i[0], axis=1, inplace=True)
        
        with self.output()['data'].open('w') as f:
            data.to_csv(f, index=False, encoding='utf-8')

# Task 5 模型
# TODO
class Model(luigi.Task):
    # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名
    num_of_feature = luigi.IntParameter(default=15)


    def requires(self):
        return FeatureSelection(is_need=self.is_need, type_of_dataset=self.type_of_dataset, name_of_dataset=self.name_of_dataset,
        name_of_target=self.name_of_target, type_of_encoding=self.type_of_encoding, cols_num=self.cols_num,num_of_feature=self.num_of_feature)
    
    def output(self):
        return {'model' : luigi.LocalTarget('Model/{0}/{1}/{1}_{2}.pkl'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model),
        format=luigi.format.Nop),
        'log' : luigi.LocalTarget('Model/{0}/{1}/{1}_{2}_log.txt'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model)),
        'train_X': luigi.LocalTarget('Model/{0}/{1}/{1}_{2}_train_X.csv'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model)),
        'test_X': luigi.LocalTarget('Model/{0}/{1}/{1}_{2}_test_X.csv'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model)),
        'test_y': luigi.LocalTarget('Model/{0}/{1}/{1}_{2}_test_Y.csv'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model))}

    def run(self):
        data = pd.read_csv(self.input()['data'].path)
        model_instance = Model_train(data, self.name_of_target, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_model=self.name_of_model)
        if self.name_of_model == 'bayesianridge':
            train, test_result, train_X,test_X,test_y = model_instance.BRreg()
        elif self.name_of_model == 'lasso':
            train, test_result, train_X,test_X,test_y = model_instance.Lreg()
        elif self.name_of_model =='gradientboosting':
            train, test_result, train_X,test_X,test_y = model_instance.GBreg()
        elif self.name_of_model =='bagging':
            train, test_result, train_X,test_X,test_y= model_instance.Breg()
        elif self.name_of_model =='xgboost':
            train, test_result, train_X,test_X,test_y = model_instance.XGBreg()
        else:
            pass
        
        with self.output()['model'].open('wb') as f:
            joblib.dump(train, f)
        with self.output()['log'].open('w') as f:
            for key, values in test_result.items():
                f.write(key + '  ' + str(round(values, 3)) + '\n')
        with self.output()['train_X'].open('w') as f:
            train_X.to_csv(f,index=False)
        with self.output()['test_X'].open('w') as f:
            test_X.to_csv(f,index=False)
        with self.output()['test_y'].open('w') as f:
            test_y.to_csv(f,index=False)

# 模型训练子任务
class BRreg(Model):
        # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名
    num_of_feature = luigi.IntParameter(default=15)

    # 模型参数
    n_iter = luigi.IntParameter(default=300)
    tol = luigi.FloatParameter(default=1e-3)
    alpha_1 = luigi.FloatParameter(default=1e-6)
    alpha_2 = luigi.FloatParameter(default=1e-6)
    lambda_1 = luigi.FloatParameter(default=1e-6)
    lambda_2 = luigi.FloatParameter(default=1e-6)
    alpha_init = luigi.FloatParameter(default=None)
    lambda_init = luigi.FloatParameter(default=None)
    compute_score = luigi.BoolParameter(default=False)
    fit_intercept = luigi.BoolParameter(default=True)
    normalize = luigi.BoolParameter(default=False)
    copy_X = luigi.BoolParameter(default=True)
    verbose = luigi.BoolParameter(default=False)

    def run(self):
        data = pd.read_csv(self.input()['data'].path)
        model_instance = Model_train(data, self.name_of_target, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_model=self.name_of_model)
        
        train, test_result, train_X,test_X,test_y = model_instance.BRreg(n_iter=self.n_iter, tol=self.tol, alpha_1=self.alpha_1, alpha_2=self.alpha_2,
        lambda_1=self.lambda_1, lambda_2=self.lambda_2, alpha_init=self.alpha_init, lambda_init=self.lambda_init, compute_score=self.compute_score,
        fit_intercept=self.fit_intercept, normalize=self.normalize, copy_X=self.copy_X, verbose=self.verbose)
        
        with super(BRreg, self).output()['model'].open('wb') as f:
            joblib.dump(train, f)
        
        with super(BRreg, self).output()['log'].open('w') as f:
            for key, values in test_result.items():
                f.write(key + '  ' + str(round(values, 3)) + '\n')
        with super(BRreg, self).output()['train_X'].open('w') as f:
            train_X.to_csv(f, index=False)
        with super(BRreg, self).output()['test_X'].open('w') as f:
            test_X.to_csv(f,index=False)
        with super(BRreg, self).output()['test_y'].open('w') as f:
            test_y.to_csv(f,index=False)

class Lreg(Model):
        # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名
    
    # 模型参数
    alpha = luigi.FloatParameter(default=1.0)
    fit_intercept = luigi.BoolParameter(default=True)
    normalize = luigi.BoolParameter(default=False)
    copy_X = luigi.BoolParameter(default=True)
    max_iter = luigi.IntParameter(default=300)
    tol = luigi.FloatParameter(default=0.001)
    random_state = luigi.IntParameter(default=None)
    selection = luigi.Parameter(default='cyclic')

    def run(self):
        data = pd.read_csv(self.input()['data'].path)
        model_instance = Model_train(data, self.name_of_target, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_model=self.name_of_model)
        
        train, test_result, train_X,test_X,test_y= model_instance.Lreg(alpha=self.alpha, fit_intercept=self.fit_intercept, normalize=self.normalize, copy_X=self.copy_X, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state, selection=self.selection)
        
        with super(Lreg, self).output()['model'].open('wb') as f:
            joblib.dump(train, f)
        
        with super(Lreg, self).output()['log'].open('w') as f:
            for key, values in test_result.items():
                f.write(key + '  ' + str(round(values, 3)) + '\n')
        with super(Lreg, self).output()['train_X'].open('w') as f:
            train_X.to_csv(f, index=False)
        with super(Lreg, self).output()['test_X'].open('w') as f:
            test_X.to_csv(f,index=False)
        with super(Lreg, self).output()['test_y'].open('w') as f:
            test_y.to_csv(f,index=False)            
                
class GBreg(Model):
        # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名
    
    # 模型参数
    loss = luigi.Parameter(default='ls')
    learning_rate = luigi.FloatParameter(default=0.1)
    n_estimators = luigi.IntParameter(default=100)
    subsample = luigi.FloatParameter(default=1.0)
    criterion = luigi.Parameter(default='friedman_mse')
    min_samples_split = luigi.IntParameter(default=2)
    min_samples_leaf = luigi.IntParameter(default=1)
    min_weight_fraction_leaf = luigi.FloatParameter(default=0.)
    max_depth = luigi.IntParameter(default=3)
    min_impurity_decrease = luigi.FloatParameter(default=0.)
    min_impurity_split = luigi.FloatParameter(default=1e-7)
    random_state = luigi.IntParameter(default=None)
    max_features = luigi.IntParameter(default=None)
    alpha = luigi.FloatParameter(default=0.9)
    verbose = luigi.IntParameter(default=0)
    max_leaf_nodes = luigi.IntParameter(default=None)
    warm_start = luigi.BoolParameter(default=False)
    validation_fraction = luigi.FloatParameter(default=0.1)
    n_iter_no_change = luigi.IntParameter(default=None)
    tol = luigi.FloatParameter(default=1e-4)
    ccp_alpha = luigi.FloatParameter(default=0.0)

    def run(self):
        data = pd.read_csv(self.input()['data'].path)
        model_instance = Model_train(data, self.name_of_target, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_model=self.name_of_model)
        
        train, test_result,train_X,test_X,test_y = model_instance.GBreg(loss=self.loss,learning_rate=self.learning_rate,n_estimators=self.n_estimators,subsample=self.subsample,criterion=self.criterion,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,min_weight_fraction_leaf=self.min_weight_fraction_leaf,max_depth=self.max_depth,min_impurity_decrease=self.min_impurity_decrease,min_impurity_split=self.min_impurity_split,random_state=self.random_state,max_features=self.max_features,alpha=self.alpha,verbose=self.verbose,max_leaf_nodes=self.max_leaf_nodes,warm_start=self.warm_start,validation_fraction=self.validation_fraction,n_iter_no_change=self.n_iter_no_change,tol=self.tol,ccp_alpha=self.ccp_alpha)
        
        with super(GBreg, self).output()['model'].open('wb') as f:
            joblib.dump(train, f)
        
        with super(GBreg, self).output()['log'].open('w') as f:
            for key, values in test_result.items():
                f.write(key + '  ' + str(round(values, 3)) + '\n')
        with super(GBreg, self).output()['train_X'].open('w') as f:
            train_X.to_csv(f, index=False)
        with super(GBreg, self).output()['test_X'].open('w') as f:
            test_X.to_csv(f,index=False)
        with super(GBreg, self).output()['test_y'].open('w') as f:
            test_y.to_csv(f,index=False)

class Breg(Model):
        # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名
    
    # 模型参数
    n_estimators = luigi.IntParameter(default=10)
    max_samples = luigi.IntParameter(default=1.0)
    max_features = luigi.IntParameter(default=1.0)
    bootstrap = luigi.BoolParameter(default=True)
    bootstrap_features = luigi.BoolParameter(default=False)
    warm_start = luigi.BoolParameter(default=False)
    n_jobs = luigi.IntParameter(default=None)
    random_state = luigi.IntParameter(default=None)
    verbose = luigi.IntParameter(default=0)

    def run(self):
        data = pd.read_csv(self.input()['data'].path)
        model_instance = Model_train(data, self.name_of_target, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_model=self.name_of_model)
        
        train, test_result, train_X,test_X,test_y = model_instance.Breg(n_estimators=self.n_estimators,max_samples=self.max_samples,max_features=self.max_features,bootstrap=self.bootstrap,bootstrap_features=self.bootstrap_features,warm_start=self.warm_start,n_jobs=self.n_jobs,random_state=self.random_state,verbose=self.verbose)
        
        with super(Breg, self).output()['model'].open('wb') as f:
            joblib.dump(train, f)
        
        with super(Breg, self).output()['log'].open('w') as f:
            for key, values in test_result.items():
                f.write(key + '  ' + str(round(values, 3)) + '\n')
        with super(Breg, self).output()['train_X'].open('w') as f:
            train_X.to_csv(f, index=False)
        with super(Breg, self).output()['test_X'].open('w') as f:
            test_X.to_csv(f,index=False)
        with super(Breg, self).output()['test_y'].open('w') as f:
            test_y.to_csv(f,index=False)
                
class XGBreg(Model):
        # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名
    
    # 模型参数
    verbosity = luigi.IntParameter(default=1)
    eta = luigi.FloatParameter(default=0.3)
    gamma = luigi.FloatParameter(default=0.)
    max_depth = luigi.IntParameter(default=6)
    min_child_weight = luigi.FloatParameter(default=1.0)
    max_delta_step = luigi.FloatParameter(default=0.)
    subsample = luigi.FloatParameter(default=1)
    colsample_bytree = luigi.FloatParameter(default=1)
    colsample_bylevel = luigi.FloatParameter(default=1)
    colsample_bynode = luigi.FloatParameter(default=1)
    lambda_ = luigi.FloatParameter(default=1)
    alpha = luigi.FloatParameter(default=0)
    sketch_eps = luigi.FloatParameter(default=0.03)
    scale_pos_weight = luigi.FloatParameter(default=1)
    refresh_leaf = luigi.FloatParameter(default=1)
    max_leaves = luigi.IntParameter(default=0)
    max_bin = luigi.IntParameter(default=256)
    num_parallel_tree = luigi.IntParameter(default=1)

    def run(self):
        data = pd.read_csv(self.input()['data'].path)
        model_instance = Model_train(data, self.name_of_target, type_of_dataset=self.type_of_dataset,
        name_of_dataset=self.name_of_dataset, name_of_model=self.name_of_model)
        
        train, test_result, train_X,test_X,test_y = model_instance.XGBreg(verbosity=verbosity,eta=eta,gamma=gamma,max_depth=max_depth,min_child_weight=min_child_weight,max_delta_step=max_delta_step,subsample=subsample,colsample_bytree=colsample_bytree,colsample_bylevel=colsample_bylevel,colsample_bynode=colsample_bynode,lambda_=lambda_,alpha=alpha,sketch_eps=sketch_eps,scale_pos_weight=scale_pos_weight,refresh_leaf=refresh_leaf,max_leaves=max_leaves,max_bin=max_bin,num_parallel_tree=num_parallel_tree)
        
        with super(XGBreg, self).output()['model'].open('wb') as f:
            joblib.dump(train, f)
        
        with super(XGBreg, self).output()['log'].open('w') as f:
            for key, values in test_result.items():
                f.write(key + '  ' + str(round(values, 3)) + '\n')
        with super(XGBreg, self).output()['train_X'].open('w') as f:
            train_X.to_csv(f, index=False)
        with super(XGBreg, self).output()['test_X'].open('w') as f:
            test_X.to_csv(f,index=False)
        with super(XGBreg, self).output()['test_y'].open('w') as f:
            test_y.to_csv(f,index=False)
# Task 6 
# TODO 模型加载以及Prediction
class Prediction(luigi.Task):
     # 参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式

    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名

    name_of_dataset_prediction = luigi.Parameter(default='used_house_data_test_encoding')

    def requires(self):
        return Model(is_need=self.is_need, type_of_dataset=self.type_of_dataset, name_of_dataset=self.name_of_dataset,
        name_of_target=self.name_of_target, type_of_encoding=self.type_of_encoding, name_of_model = self.name_of_model,
        cols_num = self.cols_num)

    def output(self):
        pass

    def run(self):
        predictor = joblib.load(self.input()['model'].path)
        data = pd.read_csv('Dataset/{0}/{1}/{2}.csv'.format(self.type_of_dataset, self.name_of_dataset, name_of_dataset_prediction))
        result = predictor.predict(data.drop([self.name_of_target],axis=1))
        np.savetxt('./Result/result.csv',result,delimiter=',')
        pass

# Task 7 模型分析
# TODO 可解释性等
class Analysis(luigi.Task):
    #  参数
    is_need = luigi.IntParameter(default=0) # 是否需要进行手动特征工程

    type_of_dataset = luigi.Parameter(default='House') # 待处理数据集类型， 目前支持House，Car,

    name_of_dataset = luigi.Parameter(default='used_house_data_test')

    name_of_target = luigi.Parameter(default='Final_Price')

    type_of_encoding = luigi.Parameter(default='target') # 目前支持one-hot target两种编码方式
    #运行可解释时需要定义name_of_model参数
    #bagging,bayesianridge,gradientboosting,lasso,xgboost
    name_of_model = luigi.Parameter(default='xgboost') # 目前支持的模型 XGBoost, Bayesian Ridge Regression, Lasso Regression, Gradient Boosting Tree, Bagging Regressor

    cols_num = luigi.ListParameter(default=['Final_Price','Transaction_Cycle','Num_Price_Adjustment', 
    'Num_Look', 'Attention', 'Num_Browse', 'Construction_Area','Age','Storey', 'Ladder', 'Household','Year_Of_Housing',
    'Property_Rights_Time']) # 数值型变量名
    num_of_feature = luigi.IntParameter(default=15)

    def requires(self):
        return {'model':Model(is_need=self.is_need, type_of_dataset=self.type_of_dataset, name_of_dataset=self.name_of_dataset,
        name_of_target=self.name_of_target, type_of_encoding=self.type_of_encoding, name_of_model = self.name_of_model,
        cols_num = self.cols_num),
            'data':FeatureSelection(is_need=self.is_need, type_of_dataset=self.type_of_dataset, name_of_dataset=self.name_of_dataset,
        name_of_target=self.name_of_target, type_of_encoding=self.type_of_encoding, cols_num=self.cols_num,num_of_feature=self.num_of_feature)}
    
                
    
    def output(self):
        """根据实际情况定义"""
        return {'globalSurrogate':luigi.LocalTarget('Model/{0}/{1}/{1}_{2}_globalex.csv'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model))}
                # 'shap':luigi.LocalTarget('Model/{0}/{1}/{1}_{2}_shap.csv'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model))}

    def run(self):
        model = joblib.load(self.input()['model']['model'].path)
        X_train=pd.read_csv(self.input()['model']['train_X'].path)
        X_test=pd.read_csv(self.input()['model']['test_X'].path)
        y_test=pd.read_csv(self.input()['model']['test_y'].path)

        index,shap_mean_df=shap_compute(X_train,model,model_name=self.name_of_model)
        fig=plt.figure(figsize=(10,8))
        ax=fig.add_subplot(111)
        ax.barh(index,shap_mean_df['shap_mean_values'].values)
        ax.set_yticks(index)
        ax.set_yticklabels(shap_mean_df['columns_name'].values)
        ax.set_xlabel('mean(|shap_value|) average impact on model output',size=15)
        plt.savefig('Model/{0}/{1}/{1}_{2}_shap_impact.png'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model))

        userData=X_train#xgb要用这样,bagging也可行
        model = 'Model/{0}/{1}/{1}_{2}.pkl'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model)
        X=pd.read_csv(self.input()['data']['data'].path)
        global_result_list=globalSurrogate(userData,model,X)
        df_result=pd.DataFrame(global_result_list,columns=X_train.columns)
        globalex_df=pd.DataFrame({'columns_name':df_result.columns,'globalex_values':df_result.values.ravel()})
        ind=np.arange(len(df_result.columns))
        fig=plt.figure(figsize=(10,8))
        ax=fig.add_subplot(111)
        ax.barh(ind,globalex_df['globalex_values'].values)
        ax.set_yticks(index)
        ax.set_yticklabels(globalex_df['columns_name'].values)
        ax.set_xlabel('Regression coefficients',size=15)
        plt.savefig('Model/{0}/{1}/{1}_{2}_globalex.png'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model))

        # residuals_plot(model, X_test, y_test,'Model/{0}/{1}/{1}_{2}_residuals.png'.format(self.type_of_dataset, self.name_of_dataset, self.name_of_model))
        with self.output()['globalSurrogate'].open('w') as f:
            df_result.to_csv(f, index=False)        
       
# Task 10 可视化
# TODO 输出各个模块的可视化结果

class Visualizer(luigi.Task):
    # 参数
    def requires(self):
        return super().requires()

    def output(self):
        return super().output()
    
    def run(self):
        return super().run()
        


if __name__ == '__main__':
    luigi.run()
    