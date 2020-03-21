def house_generator(house_data):
	house_data['Unit_Price']=house_data['Unit_Price'].astype('float')
	for i in range(len(house_data.index)):
		yield {
			"Region":house_data.iloc[i].Region,
			"Road":house_data.iloc[i].Road,
			"Community_Name":house_data.iloc[i].Community_Name,
			"Unit_Price":house_data.iloc[i].Unit_Price,
			"Room":house_data.iloc[i].Room,
			"Hall":house_data.iloc[i].Hall,
			"Construction_Area":house_data.iloc[i].Construction_Area,
			"Age":house_data.iloc[i].Age,
			"Renovation":house_data.iloc[i].Renovation,
			"Final_Price":house_data.iloc[i].Final_Price,
            "Construction_struct":house_data.iloc[i].Construction_struct,
			"Elevator":house_data.iloc[i].Elevator,
			"Trading_Authority":house_data.iloc[i].Trading_Authority,
			"Listing_Time":house_data.iloc[i].Listing_Time,
            "Ladder_Ratio":house_data.iloc[i].Ladder_Ratio,
            'Transaction_Time':house_data.iloc[i].Transaction_Time
			}
