# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:50:05 2020

@author: Furkan
"""
import urllib.request 
import requests

import pandas as pd
data = pd.read_csv('Salons_Project_clean_v_1.csv',encoding='latin')


filename="1000.jpg"
url = data["Wikipedia Image Link"][0]
url = 'http://' + url

resp = requests.get(url)

from urllib import request
f = open('00000001.jpg', 'wb')
f.write(request.urlopen(url).read())
f.close()