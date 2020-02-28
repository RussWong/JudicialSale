import os,shap,joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from yellowbrick.regressor import ResidualsPlot
def createLabel(finalPrice, criterion):
        """
        finalPrice:价格
        criterion:分档标准（List）
        return:根据价格给出一个整数，表示其价格档次
        """
        for i in range(len(criterion) + 1):
            if i == 0:
                if(finalPrice <= criterion[i]):
                    return i
            elif i == (len(criterion)):
                if(finalPrice > criterion[(len(criterion) - 1)]):
                    return i
            else:
                if(finalPrice > criterion[i - 1] and finalPrice <= criterion[i]):
                    return i
        
def getData(userData_pred,X):
    """
        获取解释用数据集
        userData_pred:用户输入数据的预测值
        address:数据类型（房（House）、车（Car））
        return:全局解释用数据子集
        step：1.对数据集进行标记（根据价格分档）
              2.取出与userData的预测值一档的数据集返回
    """
    # path = os.getcwd() + '/Dataset/used_house_data/data_code2.csv'#这里还需要改
    # X = pd.read_csv(path)
    label = list(X['Final_Price'].quantile([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]))
    X_label = X
    # X_label 增加一列label，表示依据价格对数据的分档，相当于对finalprice做了一个小小的特征工程
    X_label['Label'] = X_label.loc[:, 'Final_Price'].apply(createLabel, criterion=label)
    subX = X_label[X_label.Label == createLabel(userData_pred, criterion=label)]
    return subX

def globalSurrogate(userData, model,X):
    """
        userData:用户输入数据,dataframe
        model:模型选择（String：bagging、xgb等）
        address:数据类型（房（house）、车（car））
    """
    model = joblib.load(model)
    feature_ratio_list=[]
    #price=model.predict(np.reshape(np.array(userData),(1,-1)))#bagging等用这行可以
    price=model.predict(userData)
    coef_list=[]
    data = getData(userData_pred=price[0], X=X)#xgb这儿多了个[0]
    X_=data.drop(['Final_Price','Label'],axis=1)
    y=data['Final_Price']
    g = linear_model.LinearRegression(fit_intercept=False)#fit_intercept=False表示不会在计算中使用截距，意思就是完全是正比例函数y=kx，不是一次函数y=kx+b
    g.fit(X_, y)
    coef_list.append(g.coef_)
    return coef_list

def shap_compute(X,model,model_name):
    if model_name == 'xgboost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # fig1=shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
        shap_mean=np.mean(abs(shap_values),axis=0)
        shap_mean_df=pd.DataFrame({'columns_name':X.columns,'shap_mean_values':shap_mean}).sort_values(by='shap_mean_values').reset_index().drop('index',axis=1)
        index=np.arange(len(X.columns))
        return index,shap_mean_df
    else:
        explainer=shap.KernelExplainer(model.predict,X)
        shap_values=explainer.shap_values(X)
        # fig1=shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
        shap_mean=np.mean(abs(shap_values),axis=0)
        shap_mean_df=pd.DataFrame({'columns_name':X.columns,'shap_mean_values':shap_mean}).sort_values(by='shap_mean_values').reset_index().drop('index',axis=1)
        index=np.arange(len(X.columns))
        return index,shap_mean_df

def residuals_plot(model, X_test, y_test,road):
    """
    param 
    model : 已训练好的模型
    X_test : 测试集数据
    y_test : 测试集标签
    """
    visualizer = ResidualsPlot(model)
    visualizer.score(X_test, y_test)
    visualizer.poof(road)

