def query_house(input_data,size):
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
			'from':0, 'size':size
		}

	doc_area={
				"gauss":{
					"Construction_Area":{
						"origin":"",
						"scale":"",
						"decay":""
						#"offset":""
					}
				}
		}
	doc_area["gauss"]['Construction_Area']['origin']=input_data['Construction_Area']
	doc_area['gauss']['Construction_Area']['scale']=20 
	doc_area['gauss']['Construction_Area']['decay']=0.9

	query['query']['function_score']['functions'].append(doc_area)
	if 'Region' in input_data.keys():
		if input_data['Region']:
			doc={'match':{
				'Region': {'query':'','boost':6}}}
			doc['match']['Region']['query']=input_data['Region']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Road' in input_data.keys():
		if input_data['Road']:
			doc={'match':{
				'Road': {'query':'','boost':6}}}
			doc['match']['Road']['query']=input_data['Road']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Community_Name' in input_data.keys():
		if input_data['Community_Name']:
			doc={'match':{
					'Community_Name': {'query':'','boost':6}}}
			doc['match']['Community_Name']['query']=input_data['Community_Name']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Construction_struct' in input_data.keys():
		if input_data['Construction_struct']:
			doc={'match':{'Construction_struct': ''}}
			doc['match']['Construction_struct']=input_data['Construction_struct']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Elevator' in input_data.keys():
		if input_data['Elevator']:
			doc={"match": {
		            "Elevator": {"query": "","boost": 1}
		        }}
			doc['match']['Elevator']['query']=input_data['Elevator']
			query['query']['function_score']['query']['bool']['should'].append(doc)
	if 'Height' in input_data.keys():
		if input_data['Height']:
			doc={"match": {
		        "Height": {"query": "","boost": 1}}}
			doc['match']['Height']['query']=input_data['Height']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Floor' in input_data.keys():
		if input_data['Floor']:
			doc={"match": {
		            "Floor": {"query": "","boost": 1}
			    }}
			doc['match']['Floor']['query']=input_data['Floor']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Room' in input_data.keys():
		if input_data['Room']:
			doc={"match": {
		            "Room": {"query": "","boost": 4}
		        }}
			doc['match']['Room']['query']=input_data['Room']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Hall' in input_data.keys():
		if input_data['Hall']:
			doc={"match": {
		            "Hall": {"query": "","boost": 2}
		        }}
			doc['match']['Hall']['query']=input_data['Hall']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Ladder_Ratio' in input_data.keys():
		if input_data['Ladder_Ratio']:
			doc={"match": {
		            "Ladder_Ratio": {"query": "","boost": 3}
		        }}
			doc['match']['Ladder_Ratio']['query']=input_data['Ladder_Ratio']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Age' in input_data.keys():
		if input_data['Age']:
			doc={"match": {
		            "Age": {"query": "","boost": 1}
		        }}
			doc['match']['Age']['query']=input_data['Age']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	if 'Trading_Authority' in input_data.keys():
		if input_data['Trading_Authority']:
			doc={"match": {
		            "Trading_Authority": {"query": "","boost": 1}
		        }}
			doc['match']['Trading_Authority']['query']=input_data['Trading_Authority']
			query['query']['function_score']['query']['bool']['should'].append(doc)

	return query