def House_Generator(raw_house_data):
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
