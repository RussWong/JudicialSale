from pydantic import BaseModel
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd

import sys
sys.path.append('../Price_System/Price_Predict/')
from main_predict import price_predict
sys.path.append('../Price_System/Similar_Search/')
from search_results import search_house

app = FastAPI()

class House(BaseModel):
    #Longitude
    #Latitude
    Region: str
    Road: str
    Community_Name: str
    House_Type: str
    Construction_Area: float
    Age: int
    Renovation: str
    Construction_struct: str 
    Ladder_Ratio: str
    Elevator: str
    Storey: float 
    Ladder: float 
    Transaction_Cycle: float = None
    Num_Look: int = None
    Attention: int = None
    Household: float =None


app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/items/{id}")
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})

@app.post("/search")
async def house_search(house: House):
    # input_data={待修改!
    #             }
    input_data = {'Region':house.Region,
                  'Road':house.Road,
                  'Community':house.Community_Name,
                  'Area':house.Construction_Area,
                  'Renovation':house.Renovation,
                  'Age':str(house.Age)+'年',
                  'Room':house.House_Type[:2],
                  'Hall':house.House_Type[-2:],
                  'Struction':house.Construction_struct,
                  'Elevator':house.Elevator
                  }
    output_path = '../../output/Price_System/Similar_Search/results/'
    database = 'house'
    size=10
    house_result = search_house(input_data=input_data,size=size,output_path=output_path,database=database)
    return house_result

@app.post("/predict")
async def predict_prediction(house: House):
    # input_data={
    #             必填:
    #             'Region':所在城区,
    #             'Road':所在路段,
    #             'Community_Name':小区名字,
    #             'House_Type':户型,如2室1厅
    #             'Construction_Area':建筑面积(平方米),
    #             'Oriented':朝向,如南/北
    #             'Age':建成年份,
    #             'Renovation':装修情况,如简装,精装...
    #             'Construction_struct':建筑结构,钢混结构/砖混结构
    #             'Storey':所在楼层,
    #             'Age':建成年份,
    #             'Ladder_Ratio':梯户比例,如一梯六户,两梯两户...
    #             'Elevator':有无电梯,如有/无
    #             'Storey':所在楼层,
    #             'Ladder':楼梯数,
    #    
    #             选填:
    #             'Transaction_Cycle':交易周期,
    #             'Num_Look':带看次数
    #             'Attention':关注人数
    #             'Household':每层几户
    #             }
    input_data = {'Region':house.Region,
                  'Road':house.Road,
                  'Community_Name':house.Community_Name,
                  'House_Type':house.House_Type,
                  'Construction_Area':house.Construction_Area,
                  'Age':house.Age,
                  'Renovation':house.Renovation,
                  'Construction_struct':house.Construction_struct,
                  'Ladder_Ratio':house.Ladder_Ratio,
                  'Elevator':house.Elevator,
                  'Storey':house.Storey,
                  'Ladder':house.Ladder,
                  'Transaction_Cycle':house.Transaction_Cycle,
                  'Num_Look':house.Num_Look,
                  'Attention':house.Attention,
                  'Household':house.Household
                  }
    
    raw_data=pd.DataFrame(input_data,index=[0])

    output_path = '../../output/Price_System/Price_Predict/results/'
    encoder_path='../../output/Price_System/Price_Predict/results/encoder.csv'
    path_standModel='../../output/Price_System/Price_Predict/model/stand.pkl'
    model_path = '../../output/Price_System/Price_Predict/model/price_predict_bagging.pkl'
    
    predict_result = price_predict(raw_data=raw_data,encoder_path=encoder_path,path_standModel=path_standModel,model_path=model_path)
    return list(predict_result)

if __name__ == '__main__':
    uvicorn.run(app, port=9050, host='0.0.0.0')

