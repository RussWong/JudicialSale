#ddt模块驱动数据的改变，即批量数据通过一个测试用例
#参考链接：https://blog.csdn.net/python222/article/details/80507887
from ddt import data,ddt,unpack
import unittest

@ddt
class test_suite (unittest.TestCase):
    global input_data
    
    input_data = [{"a":2,"b":3},{"a":4,"b":8},{"a":5,"b":7}]
    #打包input_data，每条数据间用逗号分隔
    #这样做的目的是为了能够输入多条数据执行多次测试方法
    @data(*input_data)
    def test_dict(self,a):  
        """
        a:自命名，代表被逗号分隔的每条数据
        """
        self.assertLess(a["a"],a["b"])
        print(a)  
 
if __name__ == "__main__":
    unittest.main()
