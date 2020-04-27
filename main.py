# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:15:38 2020

@author: Furkan
"""

import pandas as pd
import networkx as nx
import networkx
import itertools
from numpy import sqrt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import glob
import matplotlib.image as mpimg
import cv2



SNSP = pd.read_csv('SNSP.csv',encoding='latin')
SNSP.head()
data = pd.read_csv('Salons_Project_clean_v_1.csv',encoding='latin')
data.head()
Salons=SNSP["Salons"].unique()


RousseaueInd = []
for i in range(0,data["RousseauNet"].size):
    if data["RousseauNet"][i]== 'Rousseau':
        RousseaueInd.append(i)
RousseauNet = data["SP ID"][RousseaueInd]
RousseauNet = RousseauNet.tolist()




VoltaireInd = []
for i in range(0,data["VoltaireNet"].size):
    if data["VoltaireNet"][i]== 'Voltaire':
        VoltaireInd.append(i)
VoltaireNet = data["SP ID"][VoltaireInd]
VoltaireNet = VoltaireNet.tolist()


SNSP=SNSP[SNSP["SP ID"].isin(VoltaireNet)]
SNSP=SNSP[SNSP["SP ID"].isin(RousseauNet)]




G=nx.Graph()

for i in range(0,len(VoltaireNet)):
    G.add_node(VoltaireNet[i])
    

Edges_lists=[]


for i in range(0,len(Salons)):
    tempdf = SNSP.loc[SNSP["Salons"] == Salons[i]]
    templist=tempdf["SP ID"].values.tolist()
    Edges_lists.append(templist)


Edge_weights=[]
for i in range(0,len(Edges_lists)):
    if len(Edges_lists[i]) > 2:
        tempedges = itertools.permutations(Edges_lists[i],2)
        G.add_edges_from(tempedges,weight=i)
        Edge_weights.append(i)
    else:
        pass
    
G.remove_nodes_from(node for node, degree in dict(G.degree()).items() if degree < 2)

pos = nx.circular_layout(G)

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
#edges=itertools.permutations(VoltaireNet,2)
#G.add_edges_from(edges,color='b')

#def complete_graph_from_list(L, create_using=None):
#    G = networkx.empty_graph(len(L),create_using)
#    if len(L)>1:
#        if G.is_directed():
#            edges = itertools.permutations(L,2)
#        else:
#            edges = itertools.combinations(L,2)
#        G.add_edges_from(edges)
#    return G

#G = complete_graph_from_list(VoltaireNet)
#pos=nx.spring_layout(G)
nx.draw(G,pos,width=1,edge_color=weights,edge_cmap=plt.cm.nipy_spectral,alpha=0.6,with_labels=False)

path=''
nodelist=list(G.nodes())
files = [f for f in glob.glob(path + "*.jpg")]
gray_images=['1475','1116','1360','1296','1145','1171','1346','1605','1485','1158','1139','1638','1458','1352','1576','1164','1168','1482','1019','1018','1208','1031']
image_list = []
for file in files:
    image_list.append(file[0:4])

#%%
img = []
for i in range(len(nodelist)):
    if str(nodelist[i]) in image_list:
        if str(nodelist[i]) in gray_images:
            img.append(cv2.imread(str(nodelist[i])+".jpg"))
        else:
            img.append(mpimg.imread(str(nodelist[i])+".jpg"))
    else:
        gender = (data["GenderGroup"][data["SP ID"] == str(nodelist[i])].tolist())
        if gender[0] == 'Male':
            image=cv2.imread('man.jpg',cv2.COLOR_RGB2BGR) 
            im2 = image.copy()
            im2[:, :, 0] = image[:, :, 2]
            im2[:, :, 2] = image[:, :, 0]
            img.append(im2)
        else:
            image=cv2.imread('woman.jpg',cv2.COLOR_RGB2BGR)
            im2 = image.copy()
            im2[:, :, 0] = image[:, :, 2]
            im2[:, :, 2] = image[:, :, 0]
            img.append(im2)
    
ax=plt.gca()
fig=plt.gcf()
trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform
imsize = 0.07 # this is the image size
counter=0

for n in G.nodes():
    (x,y) = pos[n]
    xx,yy = trans((x,y)) # figure coordinates
    xa,ya = trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
    a.imshow(img[counter])
    a.set_aspect('equal')
    a.axis('off')
    counter +=1








fh=open("test.edgelist",'wb')
nx.write_edgelist(G, fh)

