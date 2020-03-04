#-*- coding:utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
import sys
import os

from flask import Flask
from flask import request, render_template

sys.path.append('../Price_System/Similar_Search/')
from search_results import search_house

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/house', methods=['GET'])
def house_form():
    return render_template('house_form.html')

@app.route('/house', methods=['POST'])
def house():
    """
    """
    userRegion = request.form['Region']
    userRoad = request.form['Road']
    userCommunity = request.form['Community']
    userFloor = request.form['Floor']
    userRoom = request.form['Room']
    userHall = request.form['Hall']
    userOriented = request.form['Oriented']
    userStruction = request.form['Struction']
    userElevator = request.form['Elevator']
    userAuthority = request.form['Authority']
    userArea = request.form['Area']
    userHeight = request.form['Height']
    userAge = request.form['Age']
    
    input_data = {'Region': userRegion, 'Road': userRoad, 'Community': userCommunity, 'Area': float(userArea), 'Floor': userFloor, 'Height': '共'+userHeight+'层', 
    'Room': userRoom+'室', 'Hall': userHall+'厅','Oriented': userOriented, 'Age': userAge+'年', 'Struction': userStruction, 'Elevator': userElevator, 
    'Authority': userAuthority}

    output_path = '../../output/Price_System/Similar_Search/results/'
    database = 'house'
    size=50
    house_result = search_house(input_data=input_data,size=size,output_path=output_path,database=database)
    return render_template('house_result.html', ans1 = house_result)


if __name__ == '__main__':
    app.run()
