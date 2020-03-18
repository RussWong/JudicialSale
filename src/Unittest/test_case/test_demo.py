#ddt模块驱动数据的改变，即批量数据通过一个测试用例
from ddt import data,ddt,unpack
import unittest

@ddt
class test_suite (unittest.TestCase):
    global input_data
    
    input_data = [{"a":2,"b":3},{"a":4,"b":8},{"a":5,"b":7}]
    @data(*input_data)   
    def test_dict(self,a):  
        self.assertLess(a["a"],a["b"])
        print(a)  
 
if __name__ == "__main__":
    unittest.main()
