import pandas as pd
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# from sklearn.preprocessing import power_transform
from scipy.stats import yeojohnson
from scipy.stats import yeojohnson_normplot
from scipy.stats import probplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import kstest
def normal_transform(data_raw:pd.DataFrame, num_cols=[], is_increasing=True, lower_bounds=None, upper_bounds=None)->pd.DataFrame:
    '''Transform the numeric columns to normal distribution
    Args:
        data_raw: Input data;dataframe
        num_cols: Numeric columns to apply the transform, applying the normal transform to all the columns by default;list
        is_increasing: whether the transform should be monotonous increasing;bool
        lower_bounds: lower_bounds of the data, e.g. {col1:bound_value1,col2:bound_value2};dict
        upper_bounds: upper_bounds of the data, e.g. {col1:bound_value1,col2:bound_value2};dict
    Returns:
        data: Transformed data,same columns as Input;dataframe
    '''
    data=copy.deepcopy(data_raw)
    if data.isnull().sum().sum()>0:
        raise ValueError('NaN values not supported.')
    ind=data.index
    if not num_cols:
        num_cols=data.columns
    for col in data[num_cols].columns:
        try:
            x=np.array(data[col].astype(float))
        except:
            raise ValueError('Numeric values supported only.')
        try:
            lower_bound=lower_bounds[col]
        except:
            lower_bound=None
        try:
            upper_bound=upper_bounds[col]
        except:
            upper_bound=None
        if not lower_bound or lower_bound<x.min():
            lower_bound=x.min()

        if not upper_bound or upper_bound>x.max():
            upper_bound=x.max()

        u_raw=x.mean()
        std_raw=x.std()
        __,p_raw=kstest(x,'norm',(u_raw,std_raw))
        
        #Box-cox Transform
        if is_increasing: 
            lmbs,probs=yeojohnson_normplot(x, 0,10)
            lmb_optimal=lmbs[probs.argmax()]
            res_yj=yeojohnson(x,lmb_optimal)
            print('The optimal lambda of yeojohnson transform: ',lmb_optimal)
        else:
            res_yj,lmb_optimal=yeojohnson(x)
            print('The optimal lambda of yeojohnson transform: ',lmb_optimal)
        u_transformed_yj=res_yj.mean()
        std_transformed_yj=res_yj.std()
        __,p_yj=kstest(np.array(res_yj),'norm',(res_yj.mean(),res_yj.std()))

        res_yj=(res_yj-u_transformed_yj)*std_raw/std_transformed_yj+u_raw
        scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
        res_yj=scaler.fit_transform(res_yj.reshape(-1, 1))#turn the 1D array to 2D array

        
        
        #Arcsin square root Transform
        if x.min()<0:
            x=x-x.min()
        max_value=x.max()
        res_as=2/math.pi*pd.Series(x).apply(lambda x:math.asin(1/max_value*(x**0.5)))
        u_transformed_as=res_as.mean()
        std_transformed_as=res_as.std()
        res_as=(res_as-u_transformed_as)*std_raw/std_transformed_as+u_raw
        __,p_as=kstest(np.array(res_as),'norm',(res_as.mean(),res_as.std()))
        res_as=scaler.fit_transform(np.array(res_as).reshape(-1, 1))#turn the 1D array to 2D array
        
        
        #compare the K-S normal test p-value of raw data, yeo-johnson-transformed data and arcsin-square-root-transformed data

        if p_raw>=p_yj and p_raw>=p_as:
            print('Raw data column %s returned'%col)
            continue
        elif p_yj>=p_raw and p_yj>=p_as:
            res=res_yj
            print('Yeo-johnson-transformed data of column %s returned'%col)
        elif p_as>=p_raw and p_as>=p_yj:
            res=res_as
            print('Arcsin-square-root-transformed data of column %s returned'%col)
        data.loc[:,col]=pd.DataFrame(res,index=ind,columns=[col])
    return data