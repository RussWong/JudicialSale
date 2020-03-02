#-*- coding:utf-8 -*-
"""
此函数为用户搜索的主入口，输入查询内容，返回查询结果
"""
import pickle
import pandas as pd

from elasticsearch import Elasticsearch

from module.query import query_house

def search_house(input_data,output_path,database):
	"""
	params:
		input_data:Json,用户输入的数据
		database:String,数据库名称
	return:
		house_result:List,根据输入的数据查询到的结果
	"""
	es = Elasticsearch(['10.119.0.94'])

	#搜索规则
	query = query_house(input_data)

	#返回搜索结果
	result = es.search(index=database,body=query)
	
	#整理结果
	total_num = result['hits']['total']
	score = []
	house_result = []
	for document in result['hits']['hits']:
		score.append(document['_score'])
		house_result.append(document['_source'])
	for i in range(len(score)):
		house_result[i]["score"]=score[i]

	#保存结果
	pickle.dump(house_result,open(output_path+'similar_house.pkl','wb'))#保存结果
	return house_result

