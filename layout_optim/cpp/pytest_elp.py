import numpy as np 
import random
from time import sleep
import math
import frucht_v3
from random import choices

t = 10

k = 20
width = height = 100.0

import networkx as nx

g = nx.DiGraph()

edges = [['a', 'b'],
         ['b', 'c'],
         ['c', 'd'],
         ['d', 'a']]

# g.add_edges_from(edges)

# replace actual edges with aux el nodes
elbl_dict = {}

for e in edges:
    print(e)
    lbl_nd = "lbl_" + e[0] + e[1]

    g.add_node(lbl_nd)
    g.add_edge(e[0], lbl_nd)
    g.add_edge(e[1], lbl_nd)
    elbl_dict[lbl_nd] = (e[0], e[1])
    # g.remove_edge(e[0], e[1])



# construct index for easy querying
nd_ind = {}
c = 0
for v in g.nodes:
    nd_ind[v] = c
    c+=1


# see which nodes are edge labels (and need to be spring-adjusted), and to which two other nodes the spring goes
elbl_pos_list = []
elbl_cnct_nds = []

c = 0
for v in g.nodes:
    g.nodes[v]['x'] = choices(range(100, 700))[0]
    g.nodes[v]['y'] = choices(range(100, 700))[0]

    g.nodes[v]['width'] = choices(range(10,30))[0]
    g.nodes[v]['height'] = choices(range(10,30))[0]

    if len(v) > 4:
        if v[0:4] == 'lbl_':
            # g.nodes[v]['e_lbl'] = 1
            elbl_pos_list.append(c)
            cnct_nodes = list(g.neighbors(v))
            elbl_cnct_nds.append([nd_ind[cnct_nodes[0]], nd_ind[cnct_nodes[1]]])
    else:
        g.nodes[v]['e_lbl'] = 0
        
    c +=1


pos_nds = np.array([(g.nodes[i]['x'],g.nodes[i]['y']) for i in g.nodes])
pos_nds = pos_nds.astype(np.float32)

dim_ar = np.array([(g.nodes[i]['width'], g.nodes[i]['height']) for i in g.nodes])
dim_ar = dim_ar.astype(np.float32)


A = nx.to_numpy_array(g)
At = A.T
A = A + At
np.clip(A, 0, 1, out = A)

# actual force calcs

c = 0
for c in range(len(elbl_pos_list)):
    lbl_pos = pos_nds[c]
    p0 = pos_nds[elbl_cnct_nds[c][0]]
    p1 = pos_nds[elbl_cnct_nds[c][1]]

    v0 = p0 - lbl_pos
    v1 = p1 - lbl_pos

    v0 = v0/np.linalg.norm(v0)
    v1 = v1/np.linalg.norm(v1)
    
    dot_product = np.dot(v0,v1)
    angle = np.arccos(dot_product)

    print(angle)
    disp_vec = v0 + v1
    disp_vec = disp_vec/np.linalg.norm(disp_vec)

    # function: smaller angle -> greater force
    # no force if angle is pi
    force_mult = np.pi - angle
    # kinda would like to have some division for low angles because it really ups those forces
    # minus just doesn't cut it
    # more important than having 0 force (stuff will oscillate out anyways)

    force_mult = 1/angle
    # maybe have to multiply by constant? 
    disp_vec *= force_mult
    

