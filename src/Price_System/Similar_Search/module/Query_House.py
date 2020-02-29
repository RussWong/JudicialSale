def Query_House(input_data):
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
