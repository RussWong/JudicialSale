from elasticsearch import Elasticsearch
import json,pickle,jieba
from elasticsearch.helpers import streaming_bulk
from elasticsearch import TransportError
import pandas as pd

class SimilaritySearch():
	def House_Generator(self,raw_house_data):
		raw_house_data=pd.read_csv('../../../Data/raw/task1_house_similarity_module.csv',sep=',')
		raw_house_data['Unit_Price']=raw_house_data['Unit_Price'].astype('float')
		for i in range(len(raw_house_data.index)):
			yield {
			"Region":raw_house_data.iloc[i].Region,
			"Road":raw_house_data.iloc[i].Road,
			"Community":raw_house_data.iloc[i].Community,
			"Unit_Price":raw_house_data.iloc[i].Unit_Price,
			"Room":raw_house_data.iloc[i].Room,
			"Floor":raw_house_data.iloc[i].Floor,
			"Height":raw_house_data.iloc[i].Height,
			"Hall":raw_house_data.iloc[i].Hall,
			"Area":raw_house_data.iloc[i].Area,
			"Age":raw_house_data.iloc[i].Age,
			"Renovation":raw_house_data.iloc[i].Renovation,
			"Final_Price":raw_house_data.iloc[i].Final_Price,
			"Struction":raw_house_data.iloc[i].Struction,
			"Elevator":raw_house_data.iloc[i].Elevator,
			"Authority":raw_house_data.iloc[i].Authority,
			"Oriented":raw_house_data.iloc[i].Oriented
			}

	def Bulk_House(self):
		create_index_body={
		  "settings": {
		  "index":{
		    "number_of_shards": 1,
		    "number_of_replicas": 1,
		    "similarity" : {
		          "my_similarity" : {
		             "type" : "DFR",
		             "basic_model" : "g",
		             "after_effect" : "l",
		             "normalization" : "h2",
		             "normalization.h2.c" : "3.0"
		            }
		        }
		     }
		},
		  "mappings":{
		    "house_ershou":{
		      "properties":{
		        "Region":{
		          "type":"text",
		          "analyzer":"whitespace",
		          "similarity":"my_similarity"
		        },
		        "Road":{
			        "type":"text",
		          "analyzer":"whitespace",
			        "similarity":"my_similarity"
			    },
		        "Community":{
		          "type":"text",
		          "analyzer":"whitespace"
			 
		        },
		        "Room":{
		          "type":"text",
			        "analyzer":"whitespace"
		        },
			    "Hall":{
		          "type":"text",
			        "analyzer":"whitespace"
		        },
			    "Unit_Price":{
			      "type":"float"
			    },
		        "Floor":{
		          "type":"keyword"
		        },
		        "Oriented":{
		          "type":"keyword"
		        },
		        "Struction":{
		          "type":"text",
		          "analyzer":"whitespace"
		        },
		        "Authority":{
		          "type":"text",
		          "analyzer":"whitespace"
		        },
		        "Elevator":{
		          "type":"keyword"
		        },
			    "Height":{
		          "type":"keyword"
		        },
		        "Area":{
		          "type":"float"
		        },
		        "Age":{
		          "type":"keyword"
		        },
		        "Renovation":{
		          "type":"text",
			      "analyzer":"whitespace"
		        },
		        "Final_Price":{
		          "type":"float"
		        }
		      }
		    }
		  }
		}
		
		try:
			es.indices.create(index='house_705',body=create_index_body)
		except TransportError as e:
			if e.error=="resource_already_exists_exception":
				pass
			else:
				raise
		for ok,result in streaming_bulk(es,self.House_Generator(),index="house_705",doc_type="house_ershou"):
			document_id = "/{}/house_ershou/{}".format("house", result["index"]["_id"])
			if not ok:
				print("error")
			else:
				print(document_id)

	def Query_House(self,input_data):
		query={
			"query":{
				"function_score":{
					"functions":[],

					'query':{
						'bool':{
							'must':[],
							'should':[]
								}
							},
					"score_mode":"multiply"
					}
				},
			'from':0, 'size':50
		}

		doc_area={
				"gauss":{
					"Area":{
						"origin":"",
						"scale":"",
						"decay":""
						#"offset":""
					}
				}
		}
		doc_area["gauss"]['Area']['origin']=input_data['Area']
		doc_area['gauss']['Area']['scale']=20 
		doc_area['gauss']['Area']['decay']=0.9

		query['query']['function_score']['functions'].append(doc_area)

		if input_data['Region']:
			doc={'match':{
				'Region': {'query':'','boost':6}}}
			doc['match']['Region']['query']=input_data['Region']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Road']:
			doc={'match':{
				'Road': {'query':'','boost':6}}}
			doc['match']['Road']['query']=input_data['Road']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Community']:
			doc={'match':{
				'Community': {'query':'','boost':6}}}
			doc['match']['Community']['query']=input_data['Community']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Struction']:
			doc={'match':{'Struction': ''}}
			doc['match']['Struction']=input_data['Struction']
			query['query']['function_score']['query']['bool']['must'].append(doc)

		if input_data['Elevator']:
			doc={"match": {
	            "Elevator": {"query": "","boost": 1}
		          }}
			doc['match']['Elevator']['query']=input_data['Elevator']
			query['query']['function_score']['query']['bool']['should'].append(doc)
		if input_data['Height']:
			doc={"match": {
	            "Height": {"query": "","boost": 1}}}
			doc['match']['Height']['query']=input_data['Height']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Floor']:
			doc={"match": {
	            "Floor": {"query": "","boost": 1}
		          }}
			doc['match']['Floor']['query']=input_data['Floor']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Room']:
			doc={"match": {
	            "Room": {"query": "","boost": 4}
	          }}
			doc['match']['Room']['query']=input_data['Room']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Hall']:
			doc={"match": {
	            "Hall": {"query": "","boost": 2}
	          }}
			doc['match']['Hall']['query']=input_data['Hall']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Oriented']:
			doc={"match": {
	            "Oriented": {"query": "","boost": 3}
	          }}
			doc['match']['Oriented']['query']=input_data['Oriented']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Age']:
			doc={"match": {
	            "Age": {"query": "","boost": 1}
	          }}
			doc['match']['Age']['query']=input_data['Age']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		if input_data['Authority']:
			doc={"match": {
	            "Authority": {"query": "","boost": 1}
	          }}
			doc['match']['Authority']['query']=input_data['Authority']
			query['query']['function_score']['query']['bool']['should'].append(doc)

		return query

	def Search_House(self,input_data):
		es=Elasticsearch(["localhost"])
		query_house=self.Query_House(input_data)
		result=es.search(index='house_705',body=query_house)
		total_num=result['hits']['total']
		score=[]
		every_result=[]
		match_score=[]
		i=0
		for document in result['hits']['hits']:
			score.append(document['_score'])
			every_result.append(document['_source'])
			print('计算第{}个搜索结果的相似度...'.format(i))
		return total_num,every_result,score

	def SimilarityRun(self,input_data,dataName):
		if dataName=='house':
			num_house,house_result,house_score=self.Search_House(input_data)
			result_house=json.dumps(house_result,ensure_ascii=False)#str类型
			score_house=json.dumps(house_score)
			for i in range(len(house_score)):
				house_result[i]["score"]=house_score[i]
			print(house_result)
			pickle.dump(house_result,open('../../../output/results/task1_similarity/similar_house.pkl','wb'))

			return house_result
if __name__ == '__main__':
	input_data={
		"Area":100,
		"Region":"徐汇",
		"Road":"徐家汇",
		"Community":"东方曼哈顿",
		"Struction":"钢混结构",
		"Height":"22层",
		"Floor":"高",
		"Room":"3室",#2
		"Hall":"2厅",#1.5
		"Elevator":"有",#1.5
		"Authority":"商品房",#2
		"Oriented":"南",#2
		"Age":"2005年"
	}	
	a=SimilaritySearch()
	#a.Bulk_House()
	a.SimilarityRun(input_data=input_data,dataName='house')
