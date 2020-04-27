# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:55:42 2020

@author: Furkan
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from numpy import sqrt
import glob
import cv2
from PIL import Image 

path = ''
files = [f for f in glob.glob(path + "*.jpg")]
img = []
gray_images=['1475','1116','1360','1296','1145','1171','1346','1605','1485','1158','1139','1638','1458','1352','1576','1164','1168','1482','1019','1018','1208','1031']
for f in files:
    if f[0:4] in gray_images:        
        img.append(cv2.imread(f))
    else:
        img.append(mpimg.imread(f))
        
        
        
N = len(files)

# generate graph
G = nx.watts_strogatz_graph(N,2,0.2)
pos=nx.spring_layout(G,k=3/sqrt(N))

# draw with images on nodes
nx.draw_networkx(G,pos,width=1,edge_color="r",alpha=0.6)
ax=plt.gca()
fig=plt.gcf()
trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform
imsize = 0.07 # this is the image size
for n in G.nodes():
    (x,y) = pos[n]
    xx,yy = trans((x,y)) # figure coordinates
    xa,ya = trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
    a.imshow(img[n])
    a.set_aspect('equal')
    a.axis('off')
plt.savefig('./save.png') 