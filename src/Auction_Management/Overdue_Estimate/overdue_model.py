#标签中0表示未逾期，1表示预期
import numpy as np
import pandas as pd 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score ,accuracy_score

data_qian_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/data_qian.csv'
data_zhong_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/data_zhong.csv'
data_hou_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/data_hou.csv'
model_path = '../../../output/Auction_Management/Overdue_Estimate/model/'

def model_qian(data_qian_path,model_path):
	'''
	参数：
		data_qian_path: 拍卖前期数据路径
		model_path: 模型保存路径
	功能：
		训练并保存拍卖前期逾期模型
	'''
	data_qian = pd.read_csv(data_qian_path)
	X_qian = data_qian.drop('前期是否逾期',axis=1)
	y_qian = data_qian['前期是否逾期']
	X_train, X_test, y_train, y_test = train_test_split(X_qian,y_qian,test_size=0.2, random_state=0)

	clf_qian = LogisticRegression()  
	clf_qian.fit(X_train, y_train)

	y_pred=clf_qian.predict(X_test)
	auc=roc_auc_score(y_test,y_pred)
	acc=accuracy_score(y_test,y_pred)

	joblib.dump(clf_qian, model_path + 'pre_overdue_detection.pkl')


def model_zhong(data_zhong_path,model_path):
	'''
	参数：
		data_zhong_path: 拍卖中期数据路径
		model_path: 模型保存路径
	功能：
		训练并保存拍卖中期逾期模型
	'''
	data_zhong = pd.read_csv(data_zhong_path)
	X_zhong = data_zhong.drop('中期是否逾期',axis=1)
	y_zhong = data_zhong['中期是否逾期']
	X_train, X_test, y_train, y_test = train_test_split(X_zhong,y_zhong,test_size=0.2, random_state=0)

	clf_zhong = LogisticRegression()  
	clf_zhong.fit(X_train, y_train)

	y_pred=clf_zhong.predict(X_test)
	auc=roc_auc_score(y_test,y_pred)
	acc=accuracy_score(y_test,y_pred)

	joblib.dump(clf_zhong, model_path + 'mid_overdue_detection.pkl')


def model_hou(data_hou_path,model_path):
	'''
	参数：
		data_hou_path: 拍卖后期数据路径
		model_path: 模型保存路径
	功能：
		训练并保存拍卖后期逾期模型
	'''
	data_hou = pd.read_csv(data_hou_path)
	X_hou = data_hou.drop('后期是否逾期',axis=1)
	y_hou = data_hou['后期是否逾期']
	X_train, X_test, y_train, y_test = train_test_split(X_hou,y_hou,test_size=0.2, random_state=0)

	clf_hou = LogisticRegression()  
	clf_hou.fit(X_train, y_train)

	y_pred=clf_hou.predict(X_test)
	auc=roc_auc_score(y_test,y_pred)
	acc=accuracy_score(y_test,y_pred)

	joblib.dump(clf_hou, model_path + 'hou_overdue_detection.pkl')

