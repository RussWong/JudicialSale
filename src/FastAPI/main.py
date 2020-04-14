from typing import Optional
import base64
import uvicorn
import pandas as pd
import numpy as np
import json
from passlib.context import CryptContext
from datetime import datetime, timedelta

import jwt
from jwt import PyJWTError

from pydantic import BaseModel

from fastapi import FastAPI, Request, Form, UploadFile, HTTPException, Depends, File
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordRequestForm, OAuth2
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from starlette.status import HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED
from starlette.responses import RedirectResponse, Response, JSONResponse
from starlette.requests import Request
   
class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return round(float(obj),2)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, time):
      return obj.__str__()
    else:
      return super(NpEncoder, self).default(obj)

import sys
sys.path.append('../Price_System/Price_Predict/')
from main_predict import price_predict
sys.path.append('../Price_System/Similar_Search/')
from search_results import search_house
sys.path.append('../Auction_Management/Overdue_Estimate/')
from overdue_classify import classify_qian
sys.path.append('../Auction_Management/Overdue_Estimate/')
from overdue_classify import classify_zhong
sys.path.append('../Auction_Management/Overdue_Estimate/')
from overdue_classify import classify_hou

SECRET_KEY = "236f8aefd210910ec66c38ef9199cd8d804100d6f03ec0d87c5055e9fb012871"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DOMAIN = "10.119.0.94"

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str = None


class User(BaseModel):
    username: str
    email: str = None
    full_name: str = None
    disabled: bool = None


class UserInDB(User):
    hashed_password: str


class OAuth2PasswordBearerCookie(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: str = None,
        scopes: dict = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        header_authorization: str = request.headers.get("Authorization")
        cookie_authorization: str = request.cookies.get("Authorization")

        header_scheme, header_param = get_authorization_scheme_param(
            header_authorization
        )
        cookie_scheme, cookie_param = get_authorization_scheme_param(
            cookie_authorization
        )

        if header_scheme.lower() == "bearer":
            authorization = True
            scheme = header_scheme
            param = header_param

        elif cookie_scheme.lower() == "bearer":
            authorization = True
            scheme = cookie_scheme
            param = cookie_param

        else:
            authorization = False

        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
                )
            else:
                return None
        return param


class BasicAuth(SecurityBase):
    def __init__(self, scheme_name: str = None, auto_error: bool = True):
        self.scheme_name = scheme_name or self.__class__.__name__
        self.auto_error = auto_error

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: str = request.headers.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "basic":
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
                )
            else:
                return None
        return param


basic_auth = BasicAuth(auto_error=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearerCookie(tokenUrl="/token")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(*, data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.get("/")
async def homepage(request: Request):
    #改成请登录
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/logout")
async def route_logout_and_remove_cookie():
    response = RedirectResponse(url="/")
    response.delete_cookie("Authorization", domain = DOMAIN)
    return response

@app.get("/login_basic")
async def login_basic(auth: BasicAuth = Depends(basic_auth)):
    if not auth:
        response = Response(headers={"WWW-Authenticate": "Basic"}, status_code=401)
        return response

    try:
        decoded = base64.b64decode(auth).decode("ascii")
        username, _, password = decoded.partition(":")
        user = authenticate_user(fake_users_db, username, password)
        if not user:
            raise HTTPException(status_code=400, detail="Incorrect email or password")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username}, expires_delta=access_token_expires
        )

        token = jsonable_encoder(access_token)

        response = RedirectResponse(url="/home")
        response.set_cookie(
            "Authorization",
            value=f"Bearer {token}",
            domain= DOMAIN,
            httponly=True,
            max_age=1800,
            expires=1800,
        )
        return response

    except:
        response = Response(headers={"WWW-Authenticate": "Basic"}, status_code=401)
        return response

@app.get("/home")
async def home_page(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/price_system")
async def price_system(request: Request, current_user: User = Depends(get_current_active_user)):#方法名对应于URLfor中的名字
    return templates.TemplateResponse("price_system.html", {"request": request})

@app.get("/price_system/house")
async def house_form(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("house_form.html", {"request": request})

@app.post("/price_system/house/file_results/")
async def create_upload_files(
    file: UploadFile = File(...)
):
    contents = await file.read()
    with open('test.csv','wb') as f:
      f.write(contents)
    #  lines = f.readlines()
    #  for line in lines:
    #    content = line.decode('utf8','ignore')
    #test.csv就是用户上传的文件，存在当前目录下
    input_data = pd.read_csv('test.csv',encoding='unicode_escape',error_bad_lines=False)
    #价格预测模块
    output_path = '../../output/Price_System/Price_Predict/results/'
    encoder_path='../../output/Price_System/Price_Predict/results/encoder.csv'
    standModel_path='../../output/Price_System/Price_Predict/model/stand.pkl'
    model_path='../../output/Price_System/Price_Predict/model/price_predict_bagging.pkl'
    model_forMissing_path='../../output/Price_System/Price_Predict/model/price_predict_xgboost.pkl'
    select_feature_path='../../output/Price_System/Price_Predict/results/feature.txt'
    column_stand_path='../../output/Price_System/Price_Predict/results/featureOrder.txt'
    cols_path='../../output/Price_System/Price_Predict/results/cols.json'
    
    predict_result_list = price_predict(raw_data=input_data, encoder_path=encoder_path, 
                                   standModel_path=standModel_path, model_path=model_path, model_forMissing_path=model_forMissing_path,
                                   select_feature_path=select_feature_path, column_stand_path=column_stand_path, 
                                   cols_path=cols_path)
    print(predict_result_list)
    price_list = [round(i,2) for i in predict_result_list]
    print(price_list)
    price = json.dumps(price_list,cls=NpEncoder)

    return {'results': price}


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
                      ):

    input_search={
          'Region': Region,
          'Road': Road,
          'Community_Name': Community_Name,
          'Construction_Area': Construction_Area,
          'Renovation': Renovation,
          'Elevator': Elevator,
          'Age': Age,
          'Room': House_Type[:2],
          'Hall': House_Type[-2:],
          'Construction_struct': Construction_struct
          }

    search_output_path = '../../output/Price_System/Similar_Search/results/'
    database = 'house_ershou'
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

@app.get("/auction_management")
async def auction_management(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("./auction_management/auction_management.html", {"request": request})


@app.get("/auction_management/overdue_estimate")
async def overdue_estimate(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("./auction_management/overdue.html", {"request": request})


@app.get("/auction_management/overdue_estimate/pre_overdue")
async def overdue_pre(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("./auction_management/pre_form.html", {"request": request})


@app.get("/auction_management/overdue_estimate/mid_overdue")
async def overdue_mid(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("./auction_management/mid_form.html", {"request": request})


@app.get("/auction_management/overdue_estimate/late_overdue")
async def overdue_late(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("./auction_management/late_form.html", {"request": request})


@app.post("/auction_management/overdue_estimate/previous_period")
async def pre_overdue_estimate(request: Request,
                          Agency_Letter: int = Form(...),
                          Ruling_Letter: int = Form(...),
                          Evaluation_Price: int = Form(...),
                          Evaluation_Report: int = Form(...),
                          Evaluation_Basedate: int = Form(...),
                          Evaluation_Validity: int = Form(...),
                          Property_Rights: int = Form(...),
                          Mortgage_Status: int = Form(...),
                          Photo: int = Form(...),
                          Video: int = Form(...),
                          Announcement_Material: int = Form(...),
                          Notice: int = Form(...),
                          Fact_Sheet: int = Form(...)
                          ):
    input_data={
            "委托辅助机构函":Agency_Letter,
            "行裁定书（图片或pdf）":Ruling_Letter,
            "评估价（选择评估价时）":Evaluation_Price,
            "评估报告（图片或pdf）":Evaluation_Report,
            "评估基准日":Evaluation_Basedate,
            "评估有效期":Evaluation_Validity,
            "财产权证（图片或pdf）":Property_Rights,
            "抵押状况":Mortgage_Status,
            "标的物照片":Photo,
            "标的物视频":Video,
            "公告素材（初稿）":Announcement_Material,
            "须知素材（初稿）":Notice,
            "标的情况表":Fact_Sheet
    }
    input_data=pd.DataFrame(input_data,index=[0])
    model_path='../../output/Auction_Management/Overdue_Estimate/model/'
    res=classify_qian(input_data,model_path)
    print(res)
    table={"1":"预计拍卖前期工作会逾期。","0":"预计拍卖前期工作不会逾期。"}
    return templates.TemplateResponse("./auction_management/pre_overdue.html", {"request": request,"result":table[str(res[0])]})

@app.post("/auction_management/overdue_estimate/mid_period")
async def mid_overdue_estimate(request: Request,
                          Post_Photo: int = Form(...),
                          Auction_Notice: int = Form(...),
                          Arrange_See: int = Form(...),
                          Start_Price: int = Form(...),
                          Previous_Overdue: int = Form(...)
                          ):
    input_data={
            "公告张贴图片":Post_Photo,
            "拍卖告知凭证/其他通知方式":Auction_Notice,
            "安排看样":Arrange_See,
            "起拍价":Start_Price,
            "前期是否逾期":Previous_Overdue
    }
    input_data=pd.DataFrame(input_data,index=[0])
    model_path='../../output/Auction_Management/Overdue_Estimate/model/'
    res=classify_zhong(input_data,model_path)
    table={"1":"预计拍卖中期工作会逾期。","0":"预计拍卖中期工作不会逾期。"}
    return templates.TemplateResponse("./auction_management/mid_overdue.html", {"request": request,"result":table[str(res[0])]})

@app.post("/auction_management/overdue_estimate/late_period")
async def late_overdue_estimate(request: Request,
                          Final_Price: int = Form(...),
                          Buyer: int = Form(...),
                          Buyer_Number: int = Form(...),
                          Confirmation: int = Form(...),
                          Payment_Deadline: int = Form(...),
                          Payment_Confirm: int = Form(...),
                          Deal_Confirmation: int = Form(...),
                          Transaction_Voucher: int = Form(...),
                          Case_Report: int = Form(...),
                          Previous_Overdue : int = Form(...),
                          Mid_Overdue : int = Form(...)
                          ):
    input_data={
            "成交价":Final_Price,
            "买受人":Buyer,
            "买受人手机号":Buyer_Number,
            "竞价成功确认书截图":Confirmation,
            "余款支付截止日":Payment_Deadline,
            "成交余款支付凭证":Payment_Confirm,
            "成交确认书":Deal_Confirmation,
            "交易凭证":Transaction_Voucher,
            "结案报告":Case_Report,
            "前期是否逾期":Previous_Overdue,
            "中期是否逾期":Mid_Overdue
    }
    input_data=pd.DataFrame(input_data,index=[0])
    model_path='../../output/Auction_Management/Overdue_Estimate/model/'
    res=classify_hou(input_data,model_path)
    table={"1":"预计拍卖后期工作会逾期。","0":"预计拍卖后期工作不会逾期。"}
    return templates.TemplateResponse("./auction_management/late_overdue.html", {"request": request,"result":table[str(res[0])]})

@app.post("/auction_management/previous_period/file_results/")
async def create_upload_files(
    file: UploadFile = File(...)
):
    contents = await file.read()
    with open('pre.csv','wb') as f:
      f.write(contents)
    input_data = pd.read_csv('pre.csv',encoding='unicode_escape',error_bad_lines=False)
    model_path='../../output/Auction_Management/Overdue_Estimate/model/'

    res_list=classify_qian(input_data,model_path)
    print(res_list)
    res = json.dumps(res_list,cls=NpEncoder)

    return {'results': res}
@app.post("/auction_management/mid_period/file_results/")
async def create_upload_files(
    file: UploadFile = File(...)
):
    contents = await file.read()
    with open('mid.csv','wb') as f:
      f.write(contents)
    input_data = pd.read_csv('mid.csv',encoding='unicode_escape',error_bad_lines=False)
    model_path='../../output/Auction_Management/Overdue_Estimate/model/'

    res_list=classify_zhong(input_data,model_path)
    print(res_list)
    res = json.dumps(res_list,cls=NpEncoder)

    return {'results': res}

@app.post("/auction_management/late_period/file_results/")
async def create_upload_files(
    file: UploadFile = File(...)
):
    contents = await file.read()
    with open('late.csv','wb') as f:
      f.write(contents)
    input_data = pd.read_csv('late.csv',encoding='unicode_escape',error_bad_lines=False)
    model_path='../../output/Auction_Management/Overdue_Estimate/model/'

    res_list=classify_hou(input_data,model_path)
    print(res_list)
    res = json.dumps(res_list,cls=NpEncoder)

    return {'results': res}
if __name__ == '__main__':
    uvicorn.run(app, port=9050, host='0.0.0.0')

