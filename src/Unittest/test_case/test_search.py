import unittest
from ddt import data,ddt,unpack
import sys
sys.path.append("../../pap-2020/src/Price_System/Similar_Search/")
from search_results import search_house
#用户搜索
f = open("../data/new_house_data_test.json", encoding='utf-8')       
input_datas = json.load(f)
# input_datas=[{'Area':100},{'Region':'浦东','Road':'陆家嘴','Area':100,'Storey':4},{'Region':'浦东','Road':'陆家嘴','Storey':4},{'Region':'浦东','Road':'三林','Area':100,'Storey':4},{'Region':'浦东','Road':'陆家嘴','Storey':4}]
output_path='./output/'
database='house'
size=10

@ddt
class TestSearch(unittest.TestCase):
    
    @data(*input_datas)
    def test_if_exist_result(self,input_data):
        results = search_house(input_data=input_data,output_path=output_path,database=database,size=size)
        self.assertGreater(len(results),0)

    @data(*input_datas)
    def test_score_greater_0(self,input_data):
        results = search_house(input_data=input_data,output_path=output_path,database=database,size=size)
            for result in results:
                self.assertGreater(result['score'],0)        
      
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSearch)
    unittest.TextTestRunner(verbosity=2).run(suite)