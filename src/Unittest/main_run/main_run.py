import os
import time
import unittest

import sys
sys.path.append('../tools/')
from HTMLTestRunner import HTMLTestRunner

sys.path.append('../common/')
from logger import my_logger

class RunAll():
    def __init__(self):

        up_file_path = "../"
        #待测试模块所在路径
        self.case_module_path = os.path.join(up_file_path,"test_files","file_list.txt")
        #测试样例所在路径
        self.case_suit_path = os.path.join(up_file_path,"test_case")
        #测试报告输出路径
        self.report_path = os.path.join(up_file_path,"report")

    def set_case_list(self):
        """
        功能：获取需要测试的模块列表
        """
        case_module_list = []
        with open(self.case_module_path,"r") as f:
            for case in f.readlines():
                if case != "" and not case.startswith("#"):
                    case_module_list.append(case.replace("\n",""))
        return case_module_list

    def set_case_suit(self,case_module_list):
        """
        功能：返回包含需要测试的模块的测试套件
        """
        case_suit = unittest.TestSuite()
        case_suit_list = []
        if len(case_module_list) > 0 :
            for case_module in case_module_list :
                print("需要测试的模块："+ case_module)#例如test_demo
                discover = unittest.defaultTestLoader.discover(self.case_suit_path,pattern=case_module+".py",top_level_dir=None)
                case_suit_list.append(discover)

            for suit_list in case_suit_list:#问题：好像只能添加一个测试用例
#                 for suit in suit_list:
#                     print(suit)
                case_suit.addTest(suit_list)
        else:
            return None

        return case_suit

    def gain_report(self,suit):
        if suit is not None:
            now_time = time.strftime("%Y_%m_%d_%H_%M_%S")
            report_name = self.report_path +"/"+ now_time + "Report.html"
            with open(report_name,"wb") as f:
                runner = HTMLTestRunner(stream=f,title="*****自动化测试报告")
                runner.run(suit)
        else :            
            report_name = ''
            my_logger.info("Have no case to test.")
        return report_name
    def run(self):
        print("测试开始")
        try :
            # 1.获取需要测试的模块名
            case_module_list = self.set_case_list()
            # 2.获取要测试的套件集
            suit = self.set_case_suit(case_module_list)
            # 3.运行测试生成测试报告
            report_name = self.gain_report(suit)
        except Exception as e:
            print(e)
        print("测试结束")

if __name__ == '__main__':
    main= RunAll()
    main.run()


