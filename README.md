# PAP 2019

├──conf
│      ├──base                             <- 提供模型超参数的配置记录
│     └──local                             <- 局部配置的特别说明
│
 ├──data
│      ├──1_raw                           <- 不可更改的原数据
│     │      └──README.md     <- 数据字典
│     │
│      ├──2_intermediate          <- 数据格式转换,分类
│     │      └──README.md     
│     │
│      ├──3_qualified                 <- 筛选出含缺失值,异常值的数据
│     │      └──README.md    
│     │
│      ├──4_encoding                <- 对处理好的数据进行编码
│     │     └──README.md     
│     │
│     └──5_feature                   <-筛选出重要性特征数据
│               └──README.md     
│   
│
├──docs                                     <- 探索性数据分析文档
│     └──README.md           
│ 
├──notebooks                           <- notebook文档存放
│     └──README.md
│
├──output   
│      ├──README.md
│     │         
│      ├── 1_model                  <- 编码数据进行拟合训练后的模型
│     │
│      ├── 2_results                 <- 模型预测所得到的结果
│     │
│     └── 3_analysis               <- 对结果的分析文档
│
├──requirements.txt              <- 编程语言及版本,系统环境,工具包
│
│
└──src                                      <- 针对该项目的源代码
          ├──README.md
         │
          ├──0_main                     <-主函数体
         │      └── ().py
         │
          ├──1_intermediate      <-人工对数据格式进行转换,并对数据进行分类
         │      └── ().py
         │
          ├──2_qualified              <-数据清洗，缺失值,异常值检测与处理
         │      └── ().py
         │ 
          ├──3_encoding             <-数据编码        
         │      └── ().py
         │
          ├──4_feature                <-重要性特征选择   
         │      └── ().py
         │    
          ├──5_model                 <- 编码数据载入模型训练
         │      └── ().py
         │
          ├──6_prediction          <- 训练好的模型进行预测
         │      └── ().py
         │
         └──7_explanation       <- 对结果进行可解释性分析
                   └── ().py