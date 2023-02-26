import numpy as np
import pandas as pd

import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from ui import *
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap,QIntValidator,QImage
from PyQt5 import QtCore

import requests


try:
    with open('similarity.npy', 'rb') as f:
        similarity = np.load(f)
        
    df = pd.read_csv('dataframe.csv')
except:
    from calculate_similarity import *

finally:
    df = pd.read_csv('dataframe.csv')
    
# values
ui = Ui_MainWindow()
i = 0
distances = []
images = []
image_i = 0
# default game image
default_img = 'https://wallpapercave.com/wp/wp10389841.jpg'

def add_image(image_path):
    try:
        image = QImage()
        image.loadFromData(requests.get(image_path).content)
        qpixmap = QPixmap(image)
    except:
        qpixmap = QPixmap(image_path)
    finally:
        ui.lblPicture.setPixmap(qpixmap)
        
        
def edit_text(s):
    s = s.split(',')
    for i in range(2,len(s),2):
        s[i] = '\n\t\t\t'+s[i]
        
    return ', '.join(s).strip()
        
        
        
def add_infos():
    global i,distances,images,image_i
    similar_game = df.iloc[distances[i][0]]
    similarity = int(100*round(distances[i][1],2))

    ui.groupBox.setEnabled(True)
    title = f"{i+1}|{len(distances)-1}:\n{similar_game['name'].title()}"
    ui.lblTitle.setText(title)
    
    developer = similar_game['developer']
    publisher = similar_game['publisher']
    developer = edit_text(developer)
    publisher = edit_text(publisher)
    
    gameInfos = f"""Similarity\t:\t{similarity}%\n
Review Score\t:\t{similar_game['review_score']}\n
Release Date\t:\t{similar_game['release_date']}\n
Developer\t:\t{developer}\n
Publisher\t:\t{publisher}"""
    ui.lblGameInfos.setText(gameInfos)
    ui.lblGameInfos.adjustSize()
    
    ui.teDescription.setText(similar_game['short_description'])
    
    add_image(similar_game['header_image'])
    
    images = similar_game['screenshots'].split(',')
    images.insert(0,similar_game['header_image'])
    images.remove('')
    image_i = 0
    
    ui.leSkip.setText(f'{i+1}')
    
    
def recommend():
    global distances,i
    
    index = df[df.name == ui.leName.text()].index[0]
    
    distances = sorted(list(enumerate(similarity[index])),reverse = True,key = lambda x: x[1])
    distances = [d for d in distances if d[0] != index and d[1]>=0.55]
    
    df['similarity'] = similarity[index]
    
    add_infos()
        
        
def check_infos():
    global i,distances
    i = 0
    distances = []
    
    name = ui.leName.text().lower()

    if name != '':            
        if len(df[df['name'] ==name]) == 0:
            display_message('Invalid Game!','Entered game does not exist in file.')
        else:
            recommend()
            
    else:
        display_message('Empty Value Error!','Enter the game name.')
        
        
        
# displaying error message
def display_message(title,message):
        msg = QMessageBox()
        msg.resize(400,400)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        
def next_game():
    global i
    if i+1<len(distances)-1:
        i+=1
        add_infos()
        
        
        
def previous_game():
    global i
    if i-1 >= 0:
        i-=1
        add_infos()
        
        
def skip():
    global i
    try:
        skip_value = int(ui.leSkip.text())
        if skip_value>0 and skip_value<=len(distances)-1:
            i = skip_value-1
            add_infos()
    except Exception as ex:
        print(ex)
        
        
        
        
def next_image():
    global images,image_i
    if image_i < len(images)-1:
        image_i+=1
    else:
        image_i = 0
        
    add_image(images[image_i])
        
        
        
        
        
def previous_image():
    global images,image_i
    if image_i-1 >= 0:
        image_i-=1
    else:
        image_i = len(images) -1
    
    add_image(images[image_i])
        
        
        
        
def clear():
    global i,distances,images,image_i,default_img
    i = 0
    distances = []
    images = []
    image_i = 0
    
    ui.lblGameInfos.setText("Similarity\n\nReview Score\n\nRelease Date\n\nDeveloper\n\nPublisher")
    ui.lblGameInfos.adjustSize()
    
    ui.leSkip.setText("1")
    ui.leName.clear()
    ui.teDescription.setText("Description")
    
    ui.lblTitle.setText("Title")
    
    ui.groupBox.setEnabled(False)
    
    ui.leName.setFocus()
    
    add_image(default_img)   