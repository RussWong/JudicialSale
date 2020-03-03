#-*- coding:utf-8 -*-
"""
此函数为管理员往elasticsearch数据库批量插入数据的主入口，输入要插入的数据集和数据库名称，返回每个成功插入的样本的ID
"""
import pandas as pd

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from elasticsearch import TransportError

from module.generator import house_generator

def bulk_house(house_data_path,database):
	"""
	params:
		house_data_path:String,源数据路径
		database:String,目标数据库名称
	return:
		None
	print:
		成功插入的数据的ID
	"""
	house_data = pd.read_csv('house_data_path')
	create_index_body = {
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
	es = Elasticsearch(['10.119.0.94'])
	try:
		es.indices.create(index=database,body=create_index_body)
	except TransportError as e:
		if e.error == "resource_already_exists_exception":
			pass
		else:
			raise
	for ok,result in streaming_bulk(es,house_generator(house_data),index=database,doc_type="house_ershou"):
		document_id = "/{}/house_ershou/{}".format(database, result["index"]["_id"])
		if not ok:
			print("error")
		else:
			print(document_id)
