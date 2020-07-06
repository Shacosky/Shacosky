from flask import Flask
import requests
import json
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"


# market_id = ["btc-clp","ltc-clp","eth-clp"]

# for i in market_id:
#     precio_por_moneda = f'https://www.buda.com/api/v2/markets/{i}/ticker'
#     response = requests.get(precio_por_moneda)
#     content = response.content
#     jsondecoded = json.loads(content)
    
    
#     timestamp = 1545730073
#     dt_object = datetime.fromtimestamp(timestamp)

#     buda = jsondecoded['ticker']
#     name = buda['market_id']
#     precio = buda['last_price']
#     print("Nombre:" + name + ", Precio: " + str(precio) + " Fecha: " + str(dt_object))
#     pass