def house_generator(house_data):
	house_data['Unit_Price']=house_data['Unit_Price'].astype('float')
	for i in range(len(house_data.index)):
		yield {
			"Region":house_data.iloc[i].Region,
			"Road":house_data.iloc[i].Road,
			"Community":house_data.iloc[i].Community,
			"Unit_Price":house_data.iloc[i].Unit_Price,
			"Room":house_data.iloc[i].Room,
			"Floor":house_data.iloc[i].Floor,
			"Height":house_data.iloc[i].Height,
			"Hall":house_data.iloc[i].Hall,
			"Area":house_data.iloc[i].Area,
			"Age":house_data.iloc[i].Age,
			"Renovation":house_data.iloc[i].Renovation,
			"Final_Price":house_data.iloc[i].Final_Price,
			"Struction":house_data.iloc[i].Struction,
			"Elevator":house_data.iloc[i].Elevator,
			"Authority":house_data.iloc[i].Authority,
			"Oriented":house_data.iloc[i].Oriented
			}
