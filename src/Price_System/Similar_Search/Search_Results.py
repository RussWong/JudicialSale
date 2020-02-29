import json
import pickle
import pandas as pd

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from elasticsearch import TransportError

from module import Query_House

#input_data : Json,用户输入的数据

def Search_House(input_data):
	es=Elasticsearch(["localhost"])
	query_house=Query_House(input_data)
	result=es.search(index='house_705',body=query_house)
	total_num=result['hits']['total']
	score=[]
	house_result=[]
	match_score=[]
	for document in result['hits']['hits']:
		score.append(document['_score'])
		house_result.append(document['_source'])
	for i in range(len(score)):
		house_result[i]["score"]=score[i]
		print(house_result)
		pickle.dump(house_result,open('../../../output/results/task1_similarity/similar_house.pkl','wb'))

	return house_result

