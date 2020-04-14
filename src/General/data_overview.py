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

UNIQUE_THRESHOLD=30

def data_type_split(data_raw,cols=None,unique_threshould=UNIQUE_THRESHOLD):
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
        if num_data.shape[0]==0:
            num_data=None
        if cat_data.shape[0]==0:
            cat_data=None
        if str_data.shape[0]==0:
            str_data=None
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
FILE_PATH = './myplot/'
SUBPATH_NUM_DIST='numeric_distribution/'
SUBPATH_CAT_DIST='categorical_distribution/'
FIGURE_SIZE = 1.4
FONT_SIZE = 1
ANNOT_SIZE=12.5
LEGEND_SIZE=12.5
MAX_BAR_NUM=15
NARROW_BAR_NUM=5
UNIQUE_COUNT_FIGURENAME='unique_count'
COR_HEATMAP_FIGURENAME='numeric data correlation'

def get_unicount(data,is_plot=True,is_save=False,figsize=FIGURE_SIZE,fontsize=FONT_SIZE,path=FILE_PATH,filename=UNIQUE_COUNT_FIGURENAME):
    '''plot the unique count for all the data columns
    Args:
        data: Input data to plot the unique count
        is_plot: Flag to decide whether to plot the figure
        is_save: Flag to decide whether to save the figure
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
    if is_plot:
        fig_unitsize=np.array([6,get_adaptive_size(data.columns.shape[0])])
        plt.figure(figsize=tuple(figsize * fig_unitsize))
        data_plt=unicount_series.sort_values(ascending=True)
        y=data_plt.values
        x=data_plt.index
        b=plt.barh(x,y)
        for rect in b:
            w=rect.get_width()
            plt.text(w,rect.get_y()+rect.get_height()/2,'%d'%(w),ha='left',va='center',fontsize=fontsize*15)
        
        plt.xlim([0,y.max()*1.08])
        plt.yticks(ticks=x,fontsize=15*fontsize)
        plt.xticks(fontsize=15*fontsize)
        plt.xlabel('unique counts',fontsize=20*fontsize)
        plt.title('Data Unique Count Statistics',fontsize=20*fontsize)
        plt.show()
    if is_save:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.tight_layout()
        plt.savefig(path+filename+'.png')
        
    return unicount_series
def plot_dist(data,cols=None,is_annot_null=False,is_save=False,figsize=FIGURE_SIZE,fontsize=FONT_SIZE,path_num=FILE_PATH+SUBPATH_NUM_DIST,path_cat=FILE_PATH+SUBPATH_CAT_DIST):
    '''plot the distribution for all the numeric and categorical columns
    Args:
        data: Input data to plot
        cols: The labels of the columns ('numeric','categorical','string')
        is_annot_null: Flag to decide whether to annotate the missing data that are not plotted in the figure
        is_save: Flag to decide whether to save the figure
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        path_num: Path to save the figure of the numeric data distribution
        path_cat: Path to save the figure of the categorical data distribution
    Returns:
        None
    '''
    data_num,data_cat,data_str,data_unknown=data_type_split(data,cols)
    if data_str:
        print('Warning: columns '+','.join(data_str.columns)+' not support!')
    if data_unknown:
        print('Warning: columns '+','.join(data_unknown.columns)+' not support!')
    for col in data_cat.columns:
        data_cat.loc[data_cat[col].notnull(),col]= data_cat.loc[data_cat[col].notnull(),col].apply(lambda x:str(x))
    plot_num_dist(data_num,is_annot_null=is_annot_null,is_save=is_save,figsize=figsize,fontsize=fontsize,path=path_num)
    plot_cat_dist(data_cat,is_annot_null=is_annot_null,is_save=is_save,figsize=figsize,fontsize=fontsize,path=path_cat)

def plot_num_dist(data,is_annot_null=False,is_save=False,figsize=FIGURE_SIZE,fontsize=FONT_SIZE,path=FILE_PATH+SUBPATH_NUM_DIST):
    '''plot the distribution for all the numeric columns
    Args:
        data: Input numeric data to plot
        is_annot_null: Flag to decide whether to annotate the missing data that are not plotted in the figure
        is_save: Flag to decide whether to save the figure
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        path: Path to save the figure
    Returns:
        None
    '''
    Num_all=data.columns.shape[0]
    if is_save:
        if not os.path.exists(path):
            os.makedirs(path)
    for i in range(Num_all):
        plt.figure(i,figsize=tuple(figsize * np.array([6.5, 6.5])))
        sns.distplot(data.iloc[:,i].dropna(),hist=True,kde=False,norm_hist=False,rug=False,vertical=False,label='distplot')
        plt.xticks(fontsize=fontsize*15)
        plt.yticks(fontsize=fontsize*15)
        plt.xlabel('value',fontsize=fontsize*20)
        plt.title('Numeric Data '+data.columns[i]+' Distribution',fontsize=fontsize*20)
        if is_annot_null:
            text = 'Counted records: ' +( str(data.iloc[:,i].dropna().shape[0]) + '\n'
            +'Uncounted records:' + str(data.iloc[:,i].isnull().sum()))
            bottom_y, top_y = plt.ylim()
            bottom_x, top_x = plt.xlim()
            plt.ylim(bottom_y, top_y * 1.2)
            axis_y = top_y*1.1
            axis_x = -0.35*(top_x-bottom_x)+top_x
            plt.text(x=axis_x, y=axis_y, s=text, fontsize=fontsize * 15)
        plt.tight_layout()
        if is_save:
            plt.savefig(path+str(data.columns[i])+'.png')
        plt.show()

def plot_cat_dist(data,is_annot_null=False,is_save=False,figsize=FIGURE_SIZE,fontsize=FONT_SIZE,path=FILE_PATH+SUBPATH_CAT_DIST):
    '''plot the distribution for all the categorical columns
    Args:
        data: Input categorical data to plot 
        is_annot_null: Flag to decide whether to annotate the missing data that are not plotted in the figure
        is_save: Flag to decide whether to save the figure
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        path: Path to save the figure
    Returns:
        None
    '''
    Num_all=data.columns.shape[0]
    if is_save:
        if not os.path.exists(path):
            os.makedirs(path)
    for i in range(Num_all):
        str_counts=data.iloc[:,i].value_counts()
        
        xlabelname='values'
        if len(str_counts)>MAX_BAR_NUM:
            str_counts=str_counts.sort_values(ascending=False).iloc[0:MAX_BAR_NUM]
            xlabelname='top %d values'%MAX_BAR_NUM
        plt.figure(i,figsize=tuple(figsize * np.array([get_adaptive_size(str_counts.shape[0]), 6.5])))
        plt.bar(str_counts.index,str_counts.values,width=0.8)
        plt.xticks(rotation=90,fontsize=fontsize*15)
        plt.yticks(fontsize=fontsize*15)
        plt.xlabel(xlabelname,fontsize=fontsize*20)
        plt.title('Categorical Data '+data.columns[i]+' Distribution',fontsize=fontsize*20)
        plt.tight_layout()
        if is_annot_null:
            text = 'Counted records: ' +( str(data.iloc[:,i].dropna().shape[0]) + '\n'
            +'Uncounted records:' + str(data.iloc[:,i].isnull().sum()))
            bottom_y, top_y = plt.ylim()
            bottom_x, top_x = plt.xlim()
            plt.ylim(bottom_y, top_y * 1.2)
            axis_y = top_y*1.1
            if str_counts.shape[0]>2:
                axis_x = (top_x - bottom_x)/str_counts.shape[0]* (-2) + top_x
            else:
                axis_x = (top_x - bottom_x)/str_counts.shape[0]* (-1) + top_x
            plt.text(x=axis_x, y=axis_y, s=text, fontsize=fontsize * 15)
        if is_save:
            plt.savefig(path+str(data.columns[i])+'.png')
        plt.show()
    
def plot_num_correlation(data,is_save=False,figsize=FIGURE_SIZE,fontsize=FONT_SIZE,annot_size=ANNOT_SIZE,legend_size=LEGEND_SIZE,path=FILE_PATH,filename=COR_HEATMAP_FIGURENAME):
    '''plot the correlation heatmap for all the numeric columns
    Args:
        data: Input numeric data to plot
        is_save: Flag to decide whether to save the figure
        figsize: Size of the figure
        fontsize: Size of the font in the figure
        annot_size: Size of the annotation in the heatmap
        leagend_size: Size of the legend in the heatmap
        path: Path to save the figure
    Returns:
        None
    '''
    if not figsize:
        figsize=(data.columns.shape[0]*1.25,data.columns.shape[0]*1.25)
    plt.figure(figsize=figsize*np.array([6.5,6.5]))
    sns.heatmap(data.corr(), annot=True, vmax=1, square=True,cmap='Blues',annot_kws={'size':annot_size,'weight':'bold', 'color':'black'},fmt=".2f")
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=legend_size)
    plt.xticks(rotation=90,fontsize=fontsize*15)
    plt.yticks(rotation=0,fontsize=fontsize*15)
    plt.xlabel('numeric columns',fontsize=fontsize*20)
    plt.title('Numeric Data Correlation Statistics',fontsize=fontsize*20)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    plt.savefig(path+filename+'.png')
    plt.show()

def get_adaptive_size(num_bar:int,low_threshold=NARROW_BAR_NUM,high_threshold=MAX_BAR_NUM)->float:
    '''Calculate the proper size for the bar plots,based on the number of the bars
    Args:
        num_bar: The number of the bars
        low_threshold: The minimum size (width/height) of the bar(h) plot
        high_threshold: The maxinum size (width/height) of the bar(h) plot
    Returns:
        res: the opitmum size,width for bar plots and height for barh plots
    '''
    if num_bar<low_threshold:
        res= 1*low_threshold
    elif num_bar<high_threshold:
        res= 1.25*(num_bar-low_threshold)+low_threshold
    else:
        res= 1.25*(high_threshold-low_threshold)+low_threshold
    return res
