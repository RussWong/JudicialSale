# 测试框架目录

* common：存放一些公共方法，比如运行日志模块logging
* logs：存放log日志
* main_run：存放主函数,通过运行主函数运行测试
* report：存放测试报告
* test_case：存放测试用例函数，例如test_demo.py
* test_file/file_list.txt：需要测试的用例名称
* tools：生成测试报告的函数
* data：存放测试数据集
## 使用方法

* 写好的测试用例文件（例如：test_demo.py）放在test_case文件夹中
* 在test_file/file_list.txt中写出需要测试用例文件名，例如test_demo，若需要批量运行测试用例，则换行写出所有测试用例文件名。
* notebook cell中```%run main_run.py```    或者终端运行 ```python main_run.py```
* 生成的测试报告以html格式存于report文件夹中