## 1 数据字典

| 特征                 | 数据类型 | 中文释义     | 备注                   |
| -------------------- | -------- | ------------ | ---------------------- |
| ID                   | Int64    | 房屋ID       |                        |
| Region               | object   | 所在城区     |                        |
| Road                 | object   | 所在路段     |                        |
| Community_Name       | object   | 小区名字     |                        |
| House_Type           | object   | 户型         |                        |
| Transaction_Time     | object   | 交易时间     |                        |
| Final_Price          | Float64  | 成交价/万元  |                        |
| Unit_Price           | Float64  | 单价/元      |                        |
| Listing_Price        | Float64  | 挂牌价/万元  |                        |
| Transaction_Cycle    | Float64  | 交易周期/期  |                        |
| Num_Price_Adjustment | Int64    | 调价次数/次  |                        |
| Num_Look             | Int64    | 带看次数/次  |                        |
| Attention            | Int64    | 关注人数/人  |                        |
| Num_Browse           | Float64  | 浏览人数/人  |                        |
| Floor                | object   | 所在楼层     |                        |
| Construction_Area    | Float64  | 建筑面积/m^2 |                        |
| Unit_Structure       | object   | 户型结构     |                        |
| Type_Structure       | object   | 建筑类型     |                        |
| Oriented             | object   | 朝向         |                        |
| Age                  | Float64  | 建成年份     |                        |
| Renovation           | object   | 装修情况     |                        |
| Construction_struct  | object   | 建筑结构     |                        |
| Ladder_Ratio         | object   | 梯户比例     |                        |
| Property_Rights_Time | Float64  | 房权年限/年  |                        |
| Elevator             | object   | 有无电梯     |                        |
| Trading_Authority    | object   | 交易权属     |                        |
| Listing_Time         | object   | 挂牌时间     |                        |
| Housing_Purposes     | object   | 房屋用途     |                        |
| House_Ownership      | object   | 房权所属     |                        |
| Year_Of_Housing      | Float64  | 房屋年限/年  |                        |
| Storey               | Float64  | 所在楼层     |                        |
| Ladder               | Float64  | 几梯/个      | 由Ladder_Ratio分解所得 |
| Household            | Float64  | 几户/家      | 由Ladder_Ratio分解所得 |
| Num_Bedroom          | Float64  | 几室/间      | 由House_Type分解所得   |
| Num_Hall             | Float64  | 几厅/间      | 由House_Type分解所得   |

## 2  数据字典规范

1. 各个特征均采用26个英文字母(区分大小写)和0-9的自然数(以具体情况是否需要而定)加上下划线"\_"组成,命名简洁明确,多个单词之间以下划线"\_"分隔.

2. 特征的命名尽量完整,在保证无歧义的情况下可以使用缩写,但是需要统一缩写形式.

3. 特征不得直接以name,time,datetime这类词命名,需要加上具体的描述性定语.

4. 数据类型必须完整准确填入表中

5. 特征释义清晰,描述具体,易理解,无歧义.数值特征需要注明单位.

6. 若有特征需要特别说明,填入相应备注当中

7. 原始数据中的所有特征均需要填入数据字典.若有特殊情况在README中明确指出.

   