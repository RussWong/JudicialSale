#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import datetime
from datetime import timedelta
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

class constraint_analysis:
    def general_isDuplicated(data_raw,cols=[]):
        '''check whether the data is duplicated column by column for each record
        Args:
            data_raw: Input dataframe
            cols: The columns wanted to examine the duplication. If not passed,the function will examine all the columns
        Returns:
            isDuplicated: Result dataframe,True if the data at the corresponding location is duplicated
        '''
        data=copy.deepcopy(data_raw)
        if not cols:
            cols=data_raw.columns
        isDuplicated=pd.DataFrame(index=data.index,columns=cols).fillna(False)
        for col in cols:
            isDuplicated.loc[data[col].duplicated(keep=False),col]=True
        return isDuplicated
    def general_isWrongTyped(data_raw,typelist):
        '''check whether the data type is correct
        Args:
            data_raw: Input dataframe
            typelist: The required types of each column,dict of {columnname:type}
        Returns:
            isWrongTyped: the columns that contian wrong-typed data
        '''
        data=copy.deepcopy(data_raw)
        isWrongTyped=pd.DataFrame(index=data.index,columns=typelist.keys())
        for col in typelist.keys():
            isWrongTyped.loc[:,col]=data[col].apply(lambda x:not(isinstance(x,typelist[col])))
        return isWrongTyped
    def numCol_isbeq(data_raw,col,ref_value):
        '''check whether numeric data is greater than or equal to a reference value
        Args:
            data_raw: Input dataframe
            cols: The columns wanted to examine
        Returns:
            res: The indexes of the records that are less than the reference value
        '''
        data=copy.deepcopy(data_raw)
        try:
            res=data[col]>=ref_value
        except TypeError:
            print('Error!Different data types!')
            res=None
        return res
    def numCol_isleq(data_raw,col,ref_value):
        '''check whether numeric data is less than or equal to a reference value
        Args:
            data_raw: Input dataframe
            cols: The columns wanted to examine
        Returns:
            res: The indexes of the records that are greater than the reference value
        '''
        data=copy.deepcopy(data_raw)
        try:
            res=data[col]<=ref_value
        except TypeError:
            print('Error!Different data types!')
            res=None
        return res
    def numCol_isneq(data_raw,col,ref_value):
        '''check whether numeric data is not equal to a reference value
        Args:
            data_raw: Input dataframe
            cols: The columns wanted to examine
        Returns:
            res: The indexes of the records that are not equal to the reference value
        '''
        data=copy.deepcopy(data_raw)
        try:
            res=data[col]!=ref_value
        except TypeError:
            print('Error!Different data types!')
            res=None
        return res
    def numCols_isbeq(data_raw,col1,col2):
        '''check whether two numeric data columns satisfy a relationship that col1 >= col2 for each record
        Args:
            data_raw: Input dataframe
            col1: One of the columns to be examined
            col1: The other column to be examined
        Returns:
            res: The indexes of the records don't satisfy the relationship that col1 >= col2
        '''
        data=copy.deepcopy(data_raw)
        try:
            res=data.apply(lambda x:x[col1]>=x[col2],axis=1)
        except TypeError:
            print('Error!Different data types!')
            res=None
        return res
    def numCols_isleq(data_raw,col1,col2):
        '''check whether two numeric data columns satisfy a relationship that col1 <= col2 for each record
        Args:
            data_raw: Input dataframe
            col1: One of the columns to be examined
            col1: The other column to be examined
        Returns:
            res: The indexes of the records don't satisfy the relationship that col1 <= col2
        '''
        data=copy.deepcopy(data_raw)
        try:
            res=data.apply(lambda x:x[col1]<=x[col2],axis=1)
        except TypeError:
            print('Error!Different data types!')
            res=None
        return res
    def numCols_isneq(data_raw,col1,col2):
        '''check whether two numeric data columns satisfy a relationship that col1 != col2 for each record
        Args:
            data_raw: Input dataframe
            col1: One of the columns to be examined
            col1: The other column to be examined
        Returns:
            res: The indexes of the records don't satisfy the relationship that col1 != col2
        '''
        data=copy.deepcopy(data_raw)
        try:
            res=data.apply(lambda x:x[col1]!=x[col2],axis=1)
        except TypeError:
            print('Error!Different data types!')
            res=None
        return res
    def catCol_hasInvalid(data_raw,cols,invalid_chars):
        '''check whether a categorical data contains invalid characters
        Args:
            data_raw: Input dataframe
            col1: The columns to be examined
        Returns:
            res: The indexes of the records that contain invalid characters
        '''
        data=copy.deepcopy(data_raw)
        res=pd.DataFrame(index=data.index,columns=cols).fillna(False)
        for char in invalid_chars:
            if not isinstance(char,str):
                res=None
                print('Error!invalid_chars should be a char list!')
                return res
        try:
            for col in cols:
                res[col]=data[col].apply(lambda x:any(char in x for char in invalid_chars))
        except TypeError:
            print('Error!Non-string record in the input!')
            res=None
        return res