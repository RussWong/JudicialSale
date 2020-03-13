#-*-coding:utf-8-*-
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import datetime
from datetime import timedelta
import sys
from enum import Enum, unique
from urllib.parse import urlparse
from pandas_profiling.utils.data_types import str_is_path
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# automatic data-type check function

def data_type_split(data_raw,cols=None,unique_threshould=30):
    '''Split the data into numeric columns, categorical columns, string columns and unknown columns
    Args:
        data_raw: Input dataframe
        cols: Labels for the columns(numeric,categorical,string)
        unique_threshould: The columns that have a unique count below this threshould will be judged as categorical even if the values are numeric
    Returns:
        num_data: Numeric columns
        cat_data: Categorical columns
        str_data: (Long) string columns
        unknown_data: Columns that are not supported for split (e.g. mixed-typed data)
    '''
    data=copy.deepcopy(data_raw)
    if unique_threshould>data.shape[0]:
        print('*********************Warning! Unique_threshould larger than the sample counts!*********************')
        unique_threshould=0
    if cols:
        num_data=data_raw.loc[:,cols['numeric']]
        cat_data=data_raw.loc[:,cols['categorical']]
        str_data=data_raw.loc[:,cols['string']]
        return num_data,cat_data,str_data,None
    else:
        num_data=pd.DataFrame()
        cat_data=pd.DataFrame()
        str_data=pd.DataFrame()
        unknown_data=pd.DataFrame()
        
        for col in data.columns:
            series=pd.Series(data[col].tolist(),name=col)
            try:
                if get_var_type(series, unique_threshould)['type'] in [Variable.TYPE_NUM, Variable.TYPE_COMPLEX, Variable.TYPE_DATE]:
                    num_data=pd.concat([num_data,series],axis=1)
                elif get_var_type(series, unique_threshould)['type'] in [Variable.TYPE_CAT, Variable.TYPE_BOOL]:
                    cat_data=pd.concat([cat_data,series],axis=1)
                elif get_var_type(series, unique_threshould)['type'] in [Variable.TYPE_URL, Variable.TYPE_PATH]:
                    str_data=pd.concat([str_data,series],axis=1)
                else:
                    unknown_data=pd.concat([unknown_data,series],axis=1)
            except:
                unknown_data=pd.concat([unknown_data,series],axis=1)
        if num_data.shape[0]==0:
            num_data=None
        if cat_data.shape[0]==0:
            cat_data=None
        if str_data.shape[0]==0:
            str_data=None
        if unknown_data.shape[0]==0:
            unknown_data=None
        return num_data,cat_data,str_data,unknown_data

@unique
class Variable(Enum):
    """The possible types of variables in the Profiling Report."""

    TYPE_CAT = "CAT"
    """A categorical variable"""

    TYPE_BOOL = "BOOL"
    """A boolean variable"""

    TYPE_NUM = "NUM"
    """A numeric variable"""

    TYPE_DATE = "DATE"
    """A date variable"""

    TYPE_URL = "URL"
    """A URL variable"""

    TYPE_PATH = "PATH"
    """Absolute files"""

    TYPE_COMPLEX = "COMPLEX"

    S_TYPE_UNSUPPORTED = "UNSUPPORTED"
    """An unsupported variable"""


# Temporary mapping
Boolean = Variable.TYPE_BOOL
Real = Variable.TYPE_NUM
Count = Variable.TYPE_NUM
Complex = Variable.TYPE_COMPLEX
Date = Variable.TYPE_DATE
Categorical = Variable.TYPE_CAT
Url = Variable.TYPE_URL
AbsolutePath = Variable.TYPE_PATH
ExistingPath = Variable.TYPE_PATH
ImagePath = Variable.TYPE_PATH
Generic = Variable.S_TYPE_UNSUPPORTED


def get_counts(series: pd.Series) -> dict:
    """Counts the values in a series (with and without NaN, distinct).
    Args:
        series: Series for which we want to calculate the values.
    Returns:
        A dictionary with the count values (with and without NaN, distinct).
    """
    value_counts_with_nan = series.value_counts(dropna=False)
    value_counts_without_nan = (
        value_counts_with_nan.reset_index().dropna().set_index("index").iloc[:, 0]
    )

    distinct_count_with_nan = value_counts_with_nan.count()
    distinct_count_without_nan = value_counts_without_nan.count()

    return {
        "value_counts": value_counts_without_nan,  # Alias
        "value_counts_with_nan": value_counts_with_nan,
        "value_counts_without_nan": value_counts_without_nan,
        "distinct_count_with_nan": distinct_count_with_nan,
        "distinct_count_without_nan": distinct_count_without_nan,
    }


def is_boolean(series: pd.Series, series_description: dict) -> bool:
    """Is the series boolean type?
    Args:
        series: Series
        series_description: Series description
    Returns:
        True is the series is boolean type in the broad sense (e.g. including yes/no, NaNs allowed).
    """
    keys = series_description["value_counts_without_nan"].keys()
    if pd.api.types.is_bool_dtype(keys):
        return True
    elif (
        1 <= series_description["distinct_count_without_nan"] <= 2
        and pd.api.types.is_numeric_dtype(series)
        and series[~series.isnull()].between(0, 1).all()
    ):
        return True
    elif 1 <= series_description["distinct_count_without_nan"] <= 4:
        unique_values = set([str(value).lower() for value in keys.values])
        accepted_combinations = [
            ["y", "n"],
            ["yes", "no"],
            ["true", "false"],
            ["t", "f"],
        ]

        if len(unique_values) == 2 and any(
            [unique_values == set(bools) for bools in accepted_combinations]
        ):
            return True

    return False


def is_numeric(series: pd.Series, series_description: dict, unique_threshold:int) -> bool:
    """Is the series numeric type?
    Args:
        series: Series
        series_description: Series description
    Returns:
        True is the series is numeric type (NaNs allowed).
    """
    return pd.api.types.is_numeric_dtype(series) and series_description[
        "distinct_count_without_nan"
    ] >= int(unique_threshold)


def is_url(series: pd.Series, series_description: dict) -> bool:
    """Is the series url type?
    Args:
        series: Series
        series_description: Series description
    Returns:
        True is the series is url type (NaNs allowed).
    """
    if series_description["distinct_count_without_nan"] > 0:
        try:
            result = series[~series.isnull()].astype(str).apply(urlparse)
            return result.apply(lambda x: all([x.scheme, x.netloc, x.path])).all()
        except ValueError:
            return False
    else:
        return False


def is_path(series, series_description) -> bool:
    if series_description["distinct_count_without_nan"] > 0:
        try:
            result = series[~series.isnull()].astype(str).apply(str_is_path)
            return result.all()
        except ValueError:
            return False
    else:
        return False


def is_date(series) -> bool:
    """Is the variable of type datetime? Throws a warning if the series looks like a datetime, but is not typed as
    datetime64.
    Args:
        series: Series
    Returns:
        True if the variable is of type datetime.
    """
    is_date_value = pd.api.types.is_datetime64_dtype(series)

    return is_date_value


def get_var_type(series: pd.Series, unique_threshold=30) -> dict:
    """Get the variable type of a series.
    Args:
        series: Series for which we want to infer the variable type.
    Returns:
        The series updated with the variable type included.
    """

    series_description = {}

    try:
        series_description = get_counts(series)

        # When the inferred type of the index is just "mixed" probably the types within the series are tuple, dict,
        # list and so on...
        if series_description[
            "value_counts_without_nan"
        ].index.inferred_type.startswith("mixed"):
            raise TypeError("Not supported mixed type")

        if series_description["distinct_count_without_nan"] == 0:
            # Empty
            var_type = Variable.S_TYPE_UNSUPPORTED
        elif is_boolean(series, series_description):
            var_type = Variable.TYPE_BOOL
        elif is_numeric(series, series_description, unique_threshold):
            var_type = Variable.TYPE_NUM
        elif is_date(series):
            var_type = Variable.TYPE_DATE
        elif is_url(series, series_description):
            var_type = Variable.TYPE_URL
        elif is_path(series, series_description) and sys.version_info[1] > 5:
            var_type = Variable.TYPE_PATH
        else:
            var_type = Variable.TYPE_CAT
    except TypeError:
        var_type = Variable.S_TYPE_UNSUPPORTED

    series_description.update({"type": var_type})

    return series_description

# plot functions
def plot_gen_unicount(data,figsize=None,fontsize=25,path='./',filename='unique_count'):
    '''plot the unique count for all the data columns
    Args:
        data: Input data to plot the unique count
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        path: Path to save the figure
        filename: Filename to save the figure
    Returns:
        None
    '''
    unicount_series=pd.Series()
    for col in data.columns:
        tempdf=data[col]
        Num_unique=tempdf.unique().shape[0]
        unicount_series[col]=Num_unique
    unicount_series=unicount_series.rename('unique_counts')
    if not figsize:
        figsize=(data.columns.shape[0]*2,25)
    fig=plt.figure(figsize=figsize)
    data_plt=unicount_series.sort_values(ascending=True)
    #     print(data_plt)
    y=data_plt.values
    x=data_plt.index
    b=plt.barh(x,y)
    for rect in b:
        w=rect.get_width()
        plt.text(w,rect.get_y()+rect.get_height()/2,'%d'%(w),ha='left',va='center',fontsize=fontsize)
    plt.xlim([0,y.max()*1.1])
    plt.yticks(ticks=x,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel('unique counts',fontsize=fontsize)
    plt.title('Data Unique Count Statistics',fontsize=fontsize)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
    plt.show()

def plot_num_dist(data,figsize=(25,25),fontsize=25,path='./numeric_data_distribution'):
    '''plot the distribution for all the numeric columns
    Args:
        data: Input numeric data to plot 
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        path: Path to save the figure
    Returns:
        None
    '''
    Num_all=data.columns.shape[0]
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(Num_all):
        plt.figure(i,figsize=figsize)
        sns.distplot(data.iloc[:,i].dropna(),hist=True,kde=False,norm_hist=False,rug=False,vertical=False,label='distplot')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('value',fontsize=fontsize)
        plt.title('Numeric Data '+data.columns[i]+' Distribution',fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(path+str(data.columns[i])+'.png')
        plt.show()

def plot_cat_dist(data,figsize=(25,25),fontsize=25,bar_limit=15,path='./categorical_data_distribution'):
    '''plot the distribution for all the categorical columns
    Args:
        data: Input categorical data to plot 
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        path: Path to save the figure
    Returns:
        None
    '''
    Num_all=data.columns.shape[0]
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(Num_all):
        plt.figure(i,figsize=figsize)
        str_counts=data.iloc[:,i].value_counts()
        xlabelname='values'
        if len(str_counts)>bar_limit:
            str_counts=str_counts.sort_values(ascending=False).iloc[0:bar_limit]
            xlabelname='top %d values'%bar_limit
        plt.bar(str_counts.index,str_counts.values)
        plt.xticks(rotation=90,fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel(xlabelname,fontsize=fontsize)
        plt.title('Categorical Data '+data.columns[i]+' Distribution',fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(path+str(data.columns[i])+'.png')
        plt.show()
    
def plot_num_correlation(data,figsize=None,fontsize=25,annot_size=15,leagend_size=20,path='./',filename='numeric data correlation'):
    '''plot the correlation heatmap for all the numeric columns
    Args:
        data: Input numeric data to plot 
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        annot_size: Size of the annotation in the heatmap
        leagend_size: Size of the legend in the heatmap
        path: Path to save the figure
    Returns:
        None
    '''
    if not figsize:
        figsize=(data.columns.shape[0]*1.5,data.columns.shape[0]*1.5)
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=True, vmax=1, square=True,cmap='Blues',annot_kws={'size':annot_size,'weight':'bold', 'color':'black'},fmt=".2f")
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=leagend_size)
    plt.xticks(rotation=90,fontsize=fontsize)
    plt.yticks(rotation=0,fontsize=fontsize)
    plt.xlabel('numeric columns',fontsize=fontsize)
    plt.title('Numeric Data Correlation Statistics',fontsize=fontsize)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
    plt.show()

