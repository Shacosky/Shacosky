import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
import csv

#OpenURL
url = requests.get('https://es.investing.com/currencies/usd-clp-historical-data',headers={'User-Agent': 'Mozilla/5.0'})

#DETERMINE FORMAT
content_page = soup(url.content,'html.parser')

fecha = []
ultimo = []
apertura = []
maximo = []
minimo = []
var = []
datos = []

containers = content_page.findAll('table', {'class':'genTbl closedTbl historicalTbl'})
for table in containers:
    for td in table.findAll('td'):
        datos.append(td.text)        

for i in range(len(datos)):
    if(i % 6 == 0):
        fecha.append(datos[i])

    if(i % 6 == 1):
        ultimo.append(datos[i])

    if(i % 6 == 2):
        apertura.append(datos[i])

    if(i % 6 == 3):
        maximo.append(datos[i])

    if(i % 6 == 4):
        minimo.append(datos[i])

    if(i % 6 == 5):
        var.append(datos[i])


df = [[fecha],[ultimo],[apertura],[maximo],[minimo],[var]]

myData = []
for i in range(len(df[0][0])):
    
    if i == 0:
        myData.append(["Fecha","Ultimo","Apertura","Maximo","Minimo","Var"])

    myData.append([df[0][0][i],df[1][0][i],df[2][0][i],df[3][0][i],df[4][0][i],df[5][0][i]])
    
myFile = open('valor_dolar.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)
     
print("Writing complete")


df = pd.read_csv("valor_dolar.csv") 
print(df)

