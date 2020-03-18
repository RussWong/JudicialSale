import unittest
from ddt import data,ddt,unpack
import sys
sys.path.append("../../pap-2020/src/Price_System/Similar_Search/")
from search_results import search_house
#用户搜索
input_datas=[{'Area':100},{'Region':'浦东','Road':'陆家嘴','Area':100,'Storey':4},{'Region':'浦东','Road':'陆家嘴','Storey':4},{'Region':'浦东','Road':'三林','Area':100,'Storey':4},{'Region':'浦东','Road':'陆家嘴','Storey':4}]
output_path='./test_results/'
database='house'
size=10

@ddt
class TestSearch(unittest.TestCase):
    
    @data(*input_datas)
    def test_if_search_result(self,input_data):
        try:
            results = search_house(input_data=input_data,output_path=output_path,database=database,size=size)
            self.assertGreater(len(results),0)
        except:
            print('第{}条数据 test_if_search_result测试出错'.format(i))
           
        
    @data(*input_datas)
    def test_search_score_greater_0(self,input_datas):
        try:
            results = search_house(input_data=input_data,output_path=output_path,database=database,size=size)
            for result in results:
                self.assertGreater(result['score'],0)        
        except:
            print('第{}条数据test_search_score_greater_0测试出错'.format(i))
            
      
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSearch)
    unittest.TextTestRunner(verbosity=2).run(suite)