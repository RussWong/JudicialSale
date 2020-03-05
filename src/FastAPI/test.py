from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import sys
sys.path.append('../Price_System/Similar_Search/')
from search_results import search_house

app = FastAPI()

class House(BaseModel):
	Region: str
	Road: str
	Community: str=None
	Area: float


app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/items/{id}")
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})

@app.post("/house")
async def house_search(house: House):
    # input_data={'Region':house.Region,'Road':house.Road,'Community':house.Community,'Area':house.Area}
    input_data = house.dict()
    output_path = '../../output/Price_System/Similar_Search/results/'
    database = 'house'
    size=10
    house_result = search_house(input_data=input_data,size=size,output_path=output_path,database=database)
    return house_result

