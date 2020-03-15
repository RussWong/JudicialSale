from pydantic import BaseModel
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import numpy as np

import sys
sys.path.append('../Price_System/Price_Predict/')
from main_predict import price_predict
sys.path.append('../Price_System/Similar_Search/')
from search_results import search_house

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/admin")
async def admin(request: Request, username: str = Form(...), password: str = Form(...)):
    print('username',username)
    print('password',password)
    if username == 'admin' and password == 'admin':
        return templates.TemplateResponse('home.html', {'request': request})
    return templates.TemplateResponse("login.html", {"request": request,'msg':'输入用户名或者密码有误！'})

@app.get("/home")
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/price_system")
async def price_system(request: Request):#方法名对应于URLfor中的名字
    return templates.TemplateResponse("price_system.html", {"request": request})

@app.get("/price_system/house")
async def house_form(request: Request):
    return templates.TemplateResponse("house_form.html", {"request": request})

@app.post("/price_system/house/results")
async def price_system(request: Request,
                      Region: str = Form(...),
                      Road: str = Form(...),
                      Community_Name: str = Form(...),
                      Construction_Area: float = Form(...),
                      Renovation: str = Form(...),
                      Age: int = Form(...),
                      House_Type: str = Form(...),
                      Elevator: str = Form(...),
                      Construction_struct: str = Form(...),
                      Storey: int = Form(...),
                      Ladder_Ratio: str = Form(...),
                      Ladder: int = Form(...),
                      Transaction_Cycle: int = Form(...),
                      Num_Look : int = Form(...),
                      Attention : int = Form(...),
                      Household: int = Form(...)
                      ):#请求参数

    input_search={
          'Region': Region,
          'Road': Road,
          'Community': Community_Name,
          'Area': Construction_Area,
          'Renovation': Renovation,
          'Elevator': Elevator,
          'Age': Age,
          'Room': House_Type[:2],
          'Hall': House_Type[-2:],
          'Struction': Construction_struct
          }

    search_output_path = '../../output/Price_System/Similar_Search/results/'
    database = 'house'
    size=10
    similar_house = search_house(input_data=input_search,size=size,output_path=search_output_path,database=database)
    
    input_price = {'Region':Region,
                   'Road':Road,
                   'Community_Name':Community_Name,
                   'House_Type':House_Type,
                   'Construction_Area':Construction_Area,
                   'Age':Age,
                   'Renovation':Renovation,
                   'Construction_struct':Construction_struct,
                   'Ladder_Ratio':Ladder_Ratio,
                   'Elevator':Elevator,
                   'Storey':Storey,
                   'Ladder':Ladder,
                   'Transaction_Cycle':Transaction_Cycle,
                   'Num_Look':Num_Look,
                   'Attention':Attention,
                   'Household':Household
                   }    
                   
    if input_price['Transaction_Cycle'] == -1:
      input_price['Transaction_Cycle'] = None
    if input_price['Num_Look'] == -1:
      input_price['Num_Look'] = None
    if input_price['Attention'] == -1:
      input_price['Attention'] = None
    if input_price['Household'] == -1:
      input_price['Household'] = None 

    print(input_price)
    raw_data=pd.DataFrame(input_price,index=[0])

    output_path = '../../output/Price_System/Price_Predict/results/'
    encoder_path='../../output/Price_System/Price_Predict/results/encoder.csv'
    standModel_path='../../output/Price_System/Price_Predict/model/stand.pkl'
    model_path='../../output/Price_System/Price_Predict/model/price_predict_bagging.pkl'
    model_forMissing_path='../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
    select_feature_path='../../output/Price_System/Price_Predict/results/feature.txt'
    column_stand_path='../../output/Price_System/Price_Predict/results/featureOrder.txt'
    cols_path='../../output/Price_System/Price_Predict/results/cols.json'
    
    predict_result_list = price_predict(raw_data=raw_data, encoder_path=encoder_path, 
                                   standModel_path=standModel_path, model_path=model_path, model_forMissing_path=model_forMissing_path,
                                   select_feature_path=select_feature_path, column_stand_path=column_stand_path, 
                                   cols_path=cols_path)
    price = predict_result_list[0].round(2)

    return templates.TemplateResponse("house_results.html", {"request": request, "similar_house": similar_house, "price": price})


if __name__ == '__main__':
    uvicorn.run(app, port=9050, host='0.0.0.0')

