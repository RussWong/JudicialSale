#-*- coding:utf-8 -*-
"""
相似数据检索功能的主函数，主要包括了管理员批量插入数据功能和用户搜索功能
"""
import pandas as pd

from search_bulk import bulk_house
from search_results import search_house

#管理员批量插入数据
house_data_path='../../../Data/Price_System/Similar_Search/house.csv'
database='house'
bulk_house(house_data_path=house_data_path,database=database)

#用户搜索
input_data={}
output_path='../../../output/Price_System/Similar_Search/results/'
database='house'
search_house(input_data=input_data,output_path=output_path,database=database)