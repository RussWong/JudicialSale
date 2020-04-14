import joblib
import pandas as pd
import numpy as np 

model_path = '../../../output/Auction_Management/Overdue_Estimate/model/'

def classify_qian(input_data, model_path):
	'''
	参数：
		input_data:输入待分类的dataframe
		model_path:预测模型的path
	返回：
		result:分类结果
	'''
	clf = joblib.load(model_path+'pre_overdue_detection.pkl')
	result = clf.predict(input_data)
	result = list(result)
	return result



def classify_zhong(input_data, model_path):
	'''
	参数：
		input_data:输入待分类的dataframe
		model_path:预测模型的path
	返回：
		result:分类结果
	'''
	clf = joblib.load(model_path+'mid_overdue_detection.pkl')
	result = clf.predict(input_data)
	result = list(result)
	return result





def classify_hou(input_data, model_path):
	'''
	参数：
		input_data:输入待分类的dataframe
		model_path:预测模型的path
	返回：
		result:分类结果
	'''
	clf = joblib.load(model_path+'hou_overdue_detection.pkl')
	result = clf.predict(input_data)
	result = list(result)
	return result