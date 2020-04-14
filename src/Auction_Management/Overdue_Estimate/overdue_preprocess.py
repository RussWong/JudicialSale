import pandas as pd
import numpy as np 
import sys

data = pd.read_csv('../../../Data/Auction_Management/Overdue_Estimate/raw/拍辅通静态表.csv')
data_output_path = '../../../Data/Auction_Management/Overdue_Estimate/preprocessing/'
def preprocess_qian(data,data_output_path):
	'''
	功能：将工作前期有关字段的打分值转换为布尔值，并输出转换后的数据
	'''
	X_qian = data[['委托辅助机构函', '行裁定书（图片或pdf）', '评估价（选择评估价时）', '评估报告（图片或pdf）', '评估基准日',
	       '评估有效期', '财产权证（图片或pdf）', '抵押状况', '标的物照片', '标的物视频', '公告素材（初稿）','须知素材（初稿）', '标的情况表']]
	data['前期是否逾期'] = data['前期调查用时']
	y_qian = data['前期是否逾期']

	trans = {0.3:1,0.2:1,-0.2:0,-0.3:0}
	X_qian['委托辅助机构函'] = X_qian['委托辅助机构函'].map(trans)
	X_qian['行裁定书（图片或pdf）'] = X_qian['行裁定书（图片或pdf）'].map(trans)
	X_qian['财产权证（图片或pdf）'] = X_qian['财产权证（图片或pdf）'].map(trans)
	X_qian['标的物照片'] = X_qian['标的物照片'].map(trans)
	X_qian['标的物视频'] = X_qian['标的物视频'].map(trans)

	trans = {0.2:1,-0.2:0}
	X_qian['评估价（选择评估价时）'] = X_qian['评估价（选择评估价时）'].map(trans)
	X_qian['评估报告（图片或pdf）'] = X_qian['评估报告（图片或pdf）'].map(trans)
	X_qian['评估基准日'] = X_qian['评估基准日'].map(trans)
	X_qian['评估有效期'] = X_qian['评估有效期'].map(trans)
	X_qian['抵押状况'] = X_qian['抵押状况'].map(trans)

	trans = {0.3:1,-0.3:0}
	X_qian['公告素材（初稿）'] = X_qian['公告素材（初稿）'].map(trans)
	X_qian['须知素材（初稿）'] = X_qian['须知素材（初稿）'].map(trans)
	X_qian['标的情况表'] = X_qian['标的情况表'].map(trans)

	trans = {5:0,6:0,7:0,8:0,9:0,10:0,11:1,12:1,13:1,14:1,15:1}
	y_qian = y_qian.map(trans)
	data_qian = pd.concat([X_qian,y_qian],axis=1).to_csv(data_output_path+'data_qian.csv',index=None)


def preprocess_zhong(data,data_output_path):
	'''
	功能：将工作中期有关字段的打分值转换为布尔值，并输出转换后的数据
	'''
	data_zhong = data[['公告张贴图片','拍卖告知凭证/其他通知方式', '安排看样', '起拍价','前期调查用时','匹配时效', '公告张贴', 'EMS或其他通知方式']]
	
	trans = {0.3:1,0.2:1,-0.2:0,-0.3:0}
	data_zhong['公告张贴图片'] = data_zhong['公告张贴图片'].map(trans)
	data_zhong['拍卖告知凭证/其他通知方式'] = data_zhong['拍卖告知凭证/其他通知方式'].map(trans)
	data_zhong['起拍价'] = data_zhong['起拍价'].map(trans)
	
	trans = {'当天':1,'2天及以上':2,'未知':np.nan}
	data_zhong['匹配时效'] = data_zhong['匹配时效'].map(trans)
	data_zhong['中期用时'] = data_zhong['匹配时效']+data_zhong['公告张贴']+data_zhong['EMS或其他通知方式']
	data_zhong['中期是否逾期'] = data_zhong['中期用时'].apply(lambda x : 0 if x==13 else 1)
	
	tmp1 = data_zhong[['公告张贴图片','拍卖告知凭证/其他通知方式', '安排看样', '起拍价','前期调查用时']]
	tmp2 = data_zhong['中期是否逾期']
	X_y_zhong = pd.concat([tmp1,tmp2],axis=1).dropna(inplace=True)
	X_y_zhong['前期是否逾期'] = X_y_zhong['前期调查用时'].apply(lambda x : 0 if x<10 else 1)
	X_zhong = X_y_zhong[['公告张贴图片','拍卖告知凭证/其他通知方式', '安排看样', '起拍价','前期是否逾期']]
	y_zhong = X_y_zhong['中期是否逾期']
	
	data_zhong = pd.concat([X_zhong,y_zhong],axis=1).to_csv(data_output_path+'data_zhong.csv',index=None)


def preprocess_hou(data,data_output_path):
	'''
	功能：将工作后期有关字段的打分值转换为布尔值，并输出转换后的数据
	'''
	data_hou = data[[ '成交价', '买受人', '买受人手机号', '竞价成功确认书截图','余款支付截止日', '成交余款支付凭证', '成交确认书', '交易凭证', '结案报告']]
	
	data_hou['前期是否逾期'] = data['前期调查用时'].apply(lambda x : 0 if x<=10 else 1)
	
	trans = {'当天':1,'2天及以上':2,'未知':np.nan}
	data['匹配时效'] = data['匹配时效'].map(trans)
	data_hou['中期是否逾期'] = (data['匹配时效']+data['公告张贴']+data['EMS或其他通知方式']).apply(lambda x : 0 if x==13 else 1)
	# data_hou.drop('中期用时',axis=1,inplace=True)
	
	trans = {0.3:1,-0.3:0}
	data_hou['成交价'] = data_hou['成交价'].map(trans)
	data_hou['竞价成功确认书截图'] = data_hou['竞价成功确认书截图'].map(trans)
	data_hou['余款支付截止日'] = data_hou['余款支付截止日'].map(trans)
	data_hou['成交余款支付凭证'] = data_hou['成交余款支付凭证'].map(trans)
	data_hou['成交确认书'] = data_hou['成交确认书'].map(trans)
	data_hou['交易凭证'] = data_hou['交易凭证'].map(trans)
	
	trans = {0.2:1,-0.2:0}
	data_hou['买受人'] = data_hou['买受人'].map(trans)
	data_hou['买受人手机号'] = data_hou['买受人手机号'].map(trans)
	
	trans = {0.3:1,0.2:1,-0.3:0,-0.2:0}
	data_hou['结案报告'] = data_hou['结案报告'].map(trans)
	data_linshi = pd.read_excel('拍辅通静态表.xls')
	data_hou['后期是否逾期'] = (data_linshi['结果变更']+ data_linshi['成交后结案周期']+data_linshi['余款支付凭证上传']).apply(lambda x : 0 if x >= 1.3 else 1)
	
	X_hou = data_hou[['成交价', '买受人', '买受人手机号', '竞价成功确认书截图', '余款支付截止日', '成交余款支付凭证', '成交确认书',
	       '交易凭证', '结案报告', '前期是否逾期', '中期是否逾期']]
	y_hou = data_hou['后期是否逾期']
	
	data_hou = pd.concat([X_hou,y_hou],axis=1).to_csv(data_output_path+'data_hou.csv',index=None)
