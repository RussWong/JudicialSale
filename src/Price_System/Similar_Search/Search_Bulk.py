import pandas as pd

from elasticsearch.helpers import streaming_bulk
from elasticsearch import TransportError

from module import House_Generator

#raw_house_data :DataFrame，插入elasticsearch数据库的原始数据

def Bulk_House(raw_house_data):
	# raw_house_data=pd.read_csv('../../../Data/raw/task1_house_similarity_module.csv',sep=',')
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
	for ok,result in streaming_bulk(es,House_Generator(raw_house_data),index="house_705",doc_type="house_ershou"):
		document_id = "/{}/house_ershou/{}".format("house", result["index"]["_id"])
		if not ok:
			print("error")
		else:
			print(document_id)
