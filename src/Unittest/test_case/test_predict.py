import unittest
import pickle
import pandas as pd
from ddt import data,ddt,unpack
import sys
sys.path.append("../../Price_System/Price_Predict/")
from main_predict import price_predict

encoder_path='../../../output/Price_System/Price_Predict/results/encoder.csv'
standModel_path='../../../output/Price_System/Price_Predict/model/stand.pkl'
model_forMissing_path='../../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
select_feature_path='../../../output/Price_System/Price_Predict/results/feature.txt'
column_stand_path='../../../output/Price_System/Price_Predict/results/featureOrder.txt'
cols_path='../../../output/Price_System/Price_Predict/results/cols.json'

input_data=pd.read_csv('../../../Data/Price_System/Price_Predict/raw/house/new_house_data_test.csv')
cols = ['House_Type', 'Transaction_Cycle', 'Num_Look',
       'Attention', 'Construction_Area', 'Age', 'Renovation',
       'Construction_struct', 'Ladder_Ratio', 'Elevator', 'Storey', 'Ladder',
       'Household', 'Region', 'Road', 'Community_Name']
raw_data = input_data[cols]
raw_data_price = input_data['Final_Price']

# model_path='../../../output/Price_System/Price_Predict/model/price_predict_bagging.pkl'
# result_bagging = price_predict(raw_data, encoder_path, standModel_path, model_path, model_forMissing_path, select_feature_path, column_stand_path, cols_path)
# pickle.dump(result_bagging,open('result_bagging.pkl','wb'))
result_bagging = pickle.load(open('result_bagging.pkl','rb'))
bagging_error = []
for i in range(0,4142):
    bagging_error.append(abs((result_bagging[i]-raw_data_price[i])/raw_data_price[i]))


# model_path='../../pap-2020/output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
# result_xgboost = price_predict(raw_data, encoder_path, standModel_path, model_path, model_forMissing_path, select_feature_path, column_stand_path, cols_path)
# pickle.dump(result_xgboost,open('result_xgboost.pkl','wb'))
result_xgboost = pickle.load(open('result_xgboost.pkl','rb'))
xgboost_error = []
for i in range(0,4142):
    xgboost_error.append(abs((result_xgboost[i]-raw_data_price[i])/raw_data_price[i]))



@ddt
class test_predict(unittest.TestCase):
    
	@data(*bagging_error)    
	def test_bagging_relative_error(self,bagging_error):
		#bagging
		self.assertLess(bagging_error, 0.15)

	@data(*xgboost_error)
	def test_xgboost_relative_error(self,xgboost_error):
		#xgboost
		self.assertLess(xgboost_error, 0.15)
		
