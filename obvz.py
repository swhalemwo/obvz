#!/usr/bin/python3.9

import time

import json

from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSlot, QSize
from PyQt5 import QtCore, QtWidgets, QtGui, QtSvg


from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QFontMetrics

from time import sleep, time


from argparse import ArgumentParser
import logging


import sys
import signal

import math

import numpy as np
from random import sample, choices

from collections import Counter

import networkx as nx

from PIL import ImageFont

from graphviz import Digraph
# import graphviz


# from ovlp_func_v2 import pythran_itrtr_cbn
from frucht_v3 import frucht



app = QApplication(sys.argv)


def get_edge_point_delta(wd, ht, angle):
    """get attach point of edge for node"""

    # logging.info(['angle: ', angle])

    if angle >= 0 and angle <= math.pi/2: sector = 1
    elif angle > math.pi/2: sector = 2
    elif angle < -math.pi/2: sector = 3
    elif angle <= 0 and angle >= -math.pi/2: sector = 4

    # jfc have to define every single case separately
    if sector == 1:
        dvd_angle = math.atan(wd/ht)
        if angle < dvd_angle:
            adj, adj_type = ht/2, 'ht'
            opp = math.tan(angle)*adj
        else:
            adj, adj_type = wd/2, 'wd'
            opp = math.tan(math.pi/2 - angle)*adj

    if sector == 2:
        dvd_angle = math.pi/2 + math.atan(ht/wd)
        if angle < dvd_angle:
            adj, adj_type = wd/2, 'wd'
            opp = -math.tan(angle-math.pi/2)*adj
        else:
            adj, adj_type = -ht/2, 'ht'
            opp = abs(math.tan(math.pi - angle)*adj)

    if sector == 3:
        dvd_angle = -math.pi/2 - math.atan(ht/wd)
        if angle > dvd_angle:
            adj, adj_type = -wd/2, 'wd'
            opp = -math.tan(angle + math.pi/2)*adj
        else:
            adj, adj_type = -ht/2, 'ht'
            opp = -math.tan(-math.pi - angle)*adj

    if sector == 4:
        dvd_angle = -math.atan(wd/ht)
        if angle > dvd_angle:
            adj, adj_type = ht/2, 'ht'
            opp = math.tan(angle)*adj
        else:
            adj, adj_type = -wd/2, 'wd'
            opp = math.tan(-math.pi/2 - angle)*adj

    # print('sector: ', sector)
    # print('angle: ', angle)
    # print('dvd_angle: ', round(dvd_angle,2), round(math.degrees(dvd_angle),2))
    # print('adj: ', adj, 'adj_type: ', adj_type)
    # print('opp: ', opp)

    if adj_type == 'wd':
        dx, dy = adj, opp
    else:
        dx, dy = opp, adj
    return dx, dy

def get_reorder_order_sliced(point_nbr):
    """gets reorder sequence to re-arrange rows in point dist matrix
    now focused on extremes rather than points -> reduces number of rows to 25%
    """
    
    test_order = []
    # point_nbr = 20

    for i in range(point_nbr):
            for k in range(2):
                test_order.append(i+ (point_nbr*k))

    test_order = np.array(test_order)

    test_orders = []
    for i in range(point_nbr):
        test_orders.append((test_order + (point_nbr*2*i)).tolist())

    row_order_final = [i for sub_list in test_orders for i in sub_list]
    return row_order_final




def rect_points(r):
    p1 = [r[0] + (r[2]/2), r[1] + (r[3]/2)]
    p2 = [r[0] + (r[2]/2), r[1] - (r[3]/2)]
    p3 = [r[0] - (r[2]/2), r[1] - (r[3]/2)]
    p4 = [r[0] - (r[2]/2), r[1] + (r[3]/2)]


    return np.array([p1, p2, p3, p4])




class obvz_window(QtWidgets.QWidget):
    def __init__(self, con_type, layout_type):
        super().__init__()

        self.top= 0
        self.left= 0
        self.InitWindow()
        self.draw_arrow_toggle = 1

        # print(self.width, self.height)

        # getting font pixel information
        self.font_size = 14
        self.node_text_size = 10
        
        font_title = QFont("Arial", self.font_size)
        font_node_text = QFont("Arial", self.node_text_size)
        
        fm = QFontMetrics(font_title)
        self.title_vflush = fm.boundingRect("node title").height()

        fm = QFontMetrics(font_node_text)
        self.node_text_vflush = fm.boundingRect("node text").height()


        # self.area = self.width * self.height
        self.wd_pad = 8 # padding to add to sides of node for better drawing

        self.init_t = 12.0 # initial temperature: how much displacement possible per move, decreases
        self.def_itr = 100 # minimum number of iterations
        
        self.dt = self.init_t/(self.def_itr) # delta temperature, (int) sets number of iterations
        self.rep_nd_brd_start = 0.3 # relative size of end time frame in which node borders become repellant
        self.k = 60.0 # desired distance? 
        self.step = 8.0 # how many steps realignment takes
        self.update_interval = 40 # time for step in milliseconds
        
        # needs to be as command line parameter? think so, how else would it know how to start
        # also needs function to change it
        self.layout_type = layout_type
        self.use_edge_labels = True

        # graph information
        self.adj = [] # adjacency list
        self.node_names = []
        self.link_str = ""
        self.node_str = ""
        self.node_texts_raw = {}
        self.node_texts = {}
        self.links = []
        self.tpls = []
        self.vd = {}
        self.vdr = {}
        self.node_texts_proc = {}
        self.elbl_texts = {}
        
        self.qt_coords = np.array([])
        self.dim_ar = np.array([])

        self.g = nx.DiGraph()

        # connections
        if con_type == 'zmq':
            self.thread = QtCore.QThread()
            self.zeromq_listener = ZeroMQ_Listener()
            self.zeromq_listener.moveToThread(self.thread)
            self.thread.started.connect(self.zeromq_listener.loop)
            self.zeromq_listener.message.connect(self.signal_received)
            QtCore.QTimer.singleShot(0, self.thread.start)

        if con_type == 'dbus':
            logging.info('connection type is dbus')
            # dbus connection 
            self.bus = QtDBus.QDBusConnection.sessionBus()
            self.server = QDBusServer()
            self.bus.registerObject('/cli', self.server)
            self.bus.registerService('com.qtpad.dbus')
            self.server.dbusAdaptor.message_base.connect(self.signal_received)

        
        self.ctr = 0
        self.paint_timer = QtCore.QTimer(self, timeout=self.timer_func, interval=self.update_interval)
        

    def InitWindow(self):
        # self.setWindowTitle(self.title)
        # self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    
    def redraw_layout(self, cmd):
        """assign new rando positions"""
        if cmd == "hard":
            for n in self.g.nodes():
                
                self.g.nodes[n]['x'] = choices(range(100, self.width - 100))[0]
                self.g.nodes[n]['y'] = choices(range(100, self.height - 100))[0]


        # apply forces to current layout
        # do i even need condition? not really i think, condition is based on what calls this function
            
        self.t = self.init_t
        self.recalculate_layout()
        self.paint_timer.start()


    def reset(self):
        self.adj = [] # adjacency list? 
        self.node_names = []
        self.link_str = ""
        self.node_str = ""
        self.node_texts = {}
        self.links = []
        self.tpls = []
        self.vd = {}
        self.vdr = {}
        self.node_texts_raw = {}
        # self.node_texts_proc = {}
        self.elbl_texts = {}

        self.qt_coords = np.array([])
        self.dim_ar = np.array([])

        self.g = nx.DiGraph()
        self.update()


    def change_settings(self, setting_dict):
        """changes setting"""

        logging.info(setting_dict)
        
        setting_to_change = list(setting_dict.keys())[0]
        new_value = setting_dict[setting_to_change]

        logging.info(["setting_to_change: ", setting_to_change])
        logging.info(["new_value: ", new_value])

            
        setattr(self, setting_to_change, new_value)
        
    def cypher_query(self, qry):
        """run cypher query against the neo4j database""" 
        

        # qry ="""MATCH (t1:tag)-[:CONCERNS]->(w:work)<-[:CONCERNS]-(t2:tag)
        # WHERE t1.name >= t2.name
        # RETURN DISTINCT t1.name, t2.name, count(w) as count
        # ORDER BY count DESC LIMIT 200;"""
        logging.info(['query', qry])


        # qry = """match (col) where col.name in ["career", "cls_papers", "cls_toread", "sbcls_A", "sbcls_B", "sbcls_C", "sbcls_D", "sbcls_E"]
        # with collect(col) as cl
        # match (n {name:"career"})-[:bc]->(p)
        # with n,p,cl
        # match (f1)<-[:bp]-(p)-[:bp]->(f2) where not f1 in cl and not f2 in cl
        # with f1, f2, collect(p) as pname, count(p) as cnt
        # WHERE cnt > 2 and f1.name > f2.name
        # return f1.name, f2.name"""


        from neo4j import GraphDatabase
        
        drvr = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', 'anudora'))
        segs = drvr.session()

        res=segs.run(qry).data()

        links = [x['f1.name'] + " -- " + x['f2.name'] for x in res]
        nodes = np.unique([item for sublist in [(x['f1.name'],  x['f2.name']) for x in res] for item in sublist]).tolist()

        return {'links': links, 'nodes': nodes}
        
        
        # ## need to put res into format that can be passed to update_graph()
        # self.update_graph(new_graph_dict['links'], new_graph_dict['nodes'])
        # # self.set_node_wd_ht(list(self.g.nodes()), new_graph_dict['node_texts'])                
        # self.link_str = new_graph_dict['links']
        # self.node_str = new_graph_dict['nodes']
        # logging.info(['new self.node_str: ', self.node_str])
        # update_me = 1
        # update_texts = 1



        

    def update_visuals(self):
        """update the visuals, may have to be run often -> put into own function"""
        font_title = QFont("Arial", self.font_size)
        fm = QFontMetrics(font_title)
        self.title_vflush = fm.boundingRect("node title").height()
        self.set_node_wd_ht(list(self.g.nodes()))


    
    def signal_received(self, new_graph_str):
        """deal received signal"""
        logging.info(' receiving')
        # main parsing/updates has to happen here
        new_graph_dict = json.loads(new_graph_str)
    
        update_me = 0
        links_changed = 0
        nodes_changed = 0
        update_texts = 0

        if "draw_arrow_toggle" in new_graph_dict.keys():
            self.draw_arrow_toggle = new_graph_dict['draw_arrow_toggle']

        # change layout type
        
        
        if list(new_graph_dict.keys())[0] == "layout_type":
            if self.layout_type != new_graph_dict['layout_type']:
                self.layout_type = new_graph_dict['layout_type']
                update_me = 1
        
        if list(new_graph_dict.keys())[0] == "font_size":
            # self.font_size = new_graph_dict['font_size']
            self.change_settings(new_graph_dict)
            self.update_visuals()
            update_me = 1
            # update_texts = 1


        # only command is to redraw
        if list(new_graph_dict.keys())[0] == 'redraw':
            self.redraw_layout(new_graph_dict['redraw'])

        if list(new_graph_dict.keys())[0] == 'export':
            self.export_graph(new_graph_dict['export']['export_type'], new_graph_dict['export']['export_file'])

        
        logging.info(["graph", self.g])

        # run neo4j query
        if list(new_graph_dict.keys())[0] == 'cypher':
            neo4j_res=self.cypher_query(new_graph_dict['cypher'])
            self.update_graph(neo4j_res['links'], neo4j_res['nodes'])

            node_texts_temp = {}
            for i in neo4j_res['nodes']:
                node_texts_temp[i]="" 

            self.proc_node_texts(node_texts_temp)
                
            self.set_node_wd_ht(neo4j_res['nodes'])


            self.link_str = neo4j_res['links']
            self.node_str = neo4j_res['nodes']
            update_me = 1


        # update current node or graph
        # links and node texts as separate things
        if len(list(new_graph_dict.keys())) > 1:
            self.cur_node = new_graph_dict['cur_node']

            # toggling edge labels
            if self.use_edge_labels != new_graph_dict['use_edge_labels']:
                self.use_edge_labels = new_graph_dict['use_edge_labels']
                links_changed = 1
                update_texts = 1
                logging.info('time to update edge labels')
                
            if new_graph_dict['links'] == None:
                self.reset()

            if self.link_str != new_graph_dict['links'] and new_graph_dict['links'] != None:
                links_changed = 1
                logging.info('links have changed')
            
            if self.node_str != new_graph_dict['nodes'] and new_graph_dict['nodes'] !=  None:
                logging.info(['self.node_str: ', self.node_str])
                logging.info(["new_graph_dict['nodes']: ", new_graph_dict['nodes']])
                logging.info('nodes have changed')
                nodes_changed = 1
                
            # check if graph structure has changed
            if links_changed == 1 or nodes_changed == 1:
                logging.info('graph has changed')
                self.update_graph(new_graph_dict['links'], new_graph_dict['nodes'])
                # self.set_node_wd_ht(list(self.g.nodes()), new_graph_dict['node_texts'])                
                self.link_str = new_graph_dict['links']
                self.node_str = new_graph_dict['nodes']
                logging.info(['new self.node_str: ', self.node_str])
                update_me = 1
                update_texts = 1

            # check if node texts have been modified
            if self.node_texts_raw != new_graph_dict['node_texts'] and new_graph_dict['node_texts'] != None:
                logging.info('node texts have changed')
                logging.info(self.g.nodes())
                update_texts = 1
                
            if update_texts == 1:
                self.proc_node_texts(new_graph_dict['node_texts'])
                self.set_node_wd_ht(list(self.g.nodes()))
                
                self.node_texts_raw = new_graph_dict['node_texts']
                update_me = 1

        # start the layout calculations from here
        if update_me == 1:
            self.recalculate_layout()
            self.paint_timer.start()


        # no change: just make sure redrawing is done to take cur_node into account
        # if self.link_str == new_graph_dict['links'] and new_graph_dict['node_texts'] == self.node_texts:
        if update_me == 0:
            logging.info('graph is same, just update current node')
            self.update()

    def proc_node_texts(self, node_texts):
        """clean the node texts out of 
        - empty lines
        - links (indidcated by [[)"""

        for n in self.g.nodes():
            # ugly hack for now to get edge label nodes working
            if self.g.nodes[n]['nd_tp'] == 'nd':
                node_text_lines = [k for k in node_texts[n].split('\n') if len(k) > 0]
            else:
                node_text_lines = []
                
            node_text_lines2 = [k for k  in node_text_lines if "[[" not in k]
            logging.info(['node', n])
            logging.info(['node text lines: ', node_text_lines2])

            self.g.nodes[n]['node_text'] = node_text_lines2

            # self.node_texts_proc[i] = node_text_lines2


    def get_node_text_dimensions(self, fm_nt, node_text):
        """calculate stuff related to when text is included"""
        # n = 'bobbie'
        # node_text = new_graph_dict['node_texts'][n]
        # font = QFont("Arial", 10)
        # fm_nt = QFontMetrics(font) # font metric node text
        # ---------- test values end ----------
        
        # maybe implement some wrapping of long lines
        # but would have to determine where to wrap them, might be not straightforward with long lines

        # node_text_lines = [i for i in node_text.split('\n') if len(i) > 0]
        # get rid of links
        # node_text_lines2 = [i for i in node_text_lines if "[[" not in i]
                
        # node_rects = [fm_nt.boundingRect(i) for i in node_text_lines2]
        node_rects = [fm_nt.boundingRect(i) for i in node_text]
        widths = [i.width() for i in node_rects]
        heights = [i.height() for i in node_rects]

        return(widths, heights)


    def set_node_wd_ht(self, nodes_to_recalc_dims):
        """set height and width attributes based on text properties"""
        # hm should it be so general that i don't have to run it every time? 
        # question is if i recalculate all node properties if graph changes
        # depends on how expensive it is
        # either way should avoid multiple instantiations of fm_nt
        # also need to split up position assignment and height/width calculation 
        
        logging.info('setting attributes')

        # font = ImageFont.truetype('Arial', self.font_size)
        
        # font = QFont("Arial", 12)
        font = QFont("Arial", self.font_size)
        fm = QFontMetrics(font)


        font = QFont("Arial", self.node_text_size)
        fm_nt = QFontMetrics(font) # font metric node text


        for n in nodes_to_recalc_dims:
            if self.g.nodes[n]['nd_tp'] == 'nd':
                node_rect = fm.boundingRect(self.g.nodes[n]['title'])
                nd_title_wd, nd_title_ht = node_rect.width(), node_rect.height()
                
                # get sizes of node body text lines
                nt_dims = self.get_node_text_dimensions(fm_nt, self.g.nodes[n]['node_text'])
                nt_dims[0].append(nd_title_wd)
                nt_dims[1].append(nd_title_ht)
                
                # node_sz = (node_rect.width() + self.wd_pad*2, node_rect.height())
                # self.g.nodes[n]['width'] = node_sz[0]
                # self.g.nodes[n]['height'] = node_sz[1]

                self.g.nodes[n]['width'] = max(nt_dims[0]) + self.wd_pad*2
                self.g.nodes[n]['height'] = sum(nt_dims[1])
                logging.info(['node dims: ', self.g.nodes[n]['width'], self.g.nodes[n]['height']])

            # if node is an edge label node there's no need to check for node text
            if self.g.nodes[n]['nd_tp'] == 'lbl':
                node_rect = fm_nt.boundingRect(self.g.nodes[n]['title'])
                nd_title_wd, nd_title_ht = node_rect.width(), node_rect.height()
                self.g.nodes[n]['width'] = nd_title_wd + self.wd_pad*2
                self.g.nodes[n]['height'] = nd_title_ht
            
        self.dim_ar = np.array([[self.g.nodes[i]['width'], self.g.nodes[i]['height']] for i in self.g.nodes])




    def update_graph(self, new_links, incoming_nodes):
        
        """set new links and nodes"""

        # need clear names: 
        # "new": refers to new graph as a whole (can exist before), 
        # "old": being in old graph, irrespective of being in new one
        # "add": existing in new, not in old
        # "del": in old, not in new

        # new_links = new_link_str.split(";")

        elbl_title_dict = {}
        
        new_tpls = []
        if new_links is not None:
            # new_tpls = [(i.split(" -- ")[0], i.split(" -- ")[1]) for i in new_links]
            for i in new_links:
                tpl = i.split(" -- ")
                # build graph depending on whether edge labels are used or not
                
                if self.use_edge_labels == True: 

                    elbl_nd_name = "lbl_" + tpl[0] + "_" + tpl[1] + "_" + tpl[2]
                    new_tpls.append((tpl[0], elbl_nd_name))
                    new_tpls.append((elbl_nd_name, tpl[1]))
                    
                    elbl_title_dict[elbl_nd_name] = tpl[2]
                    self.elbl_texts[tpl[2]] = tpl[2]
                    
                else:
                    new_tpls.append((tpl[0], tpl[1]))
        else:
            new_tpls = set()

        logging.info(elbl_title_dict)

        tpls_to_add = list(set(new_tpls) - set(self.tpls))
        tpls_to_del = list(set(self.tpls) - set(new_tpls))
        
        # not clear if links (one string) are that relevant, tuples seem to more convenient to work with tbh
        # old_tpls = self.tpls
        self.tpls = new_tpls
        # self.links = new_links

        old_nodes = self.g.nodes()

        # prevent modification of original node object
        new_nodes = incoming_nodes.copy()

        for l in self.tpls:
            new_nodes.append(l[0])
            new_nodes.append(l[1])
        new_nodes = set(new_nodes)
        
        
        nodes_to_del = list(old_nodes - new_nodes)
        nodes_to_add = list(new_nodes - old_nodes)

        logging.info(['new tpls: ', new_tpls])
        # logging.info(['old tpls: ', old_tpls])

        logging.info(["links_to_add: ", tpls_to_add])
        logging.info(["links_to_del: ", tpls_to_del])

        logging.info(['new nodes: ', new_nodes])
        logging.info(['old nodes: ', old_nodes])
        
        logging.info(["nodes_to_add: ", nodes_to_add])
        logging.info(["nodes_to_del: ", nodes_to_del])
        
        # first add nodes + index them
        # actually not clear if vd is needed much
        index_pos = len(self.g.nodes)

        for n in nodes_to_add:
            logging.info('adding node')
            
            self.g.add_node(n)
            # have to add title stuff here
            if n[0:4] == "lbl_":
                self.g.nodes[n]['title'] = elbl_title_dict[n]
                self.g.nodes[n]['nd_tp'] = 'lbl'
            else:
                self.g.nodes[n]['title'] = n
                self.g.nodes[n]['nd_tp'] = 'nd'
                
            self.vd[n] = index_pos
            self.vdr[index_pos] = n
            index_pos +=1

        self.g.remove_nodes_from(nodes_to_del)
        logging.info('nodes deleted')
        # have to reindex after deletion

        self.vd = {}
        self.vdr = {}
        c = 0
        for i in self.g.nodes():
            self.vd[i] = c
            self.vdr[c] = i
            c += 1
        
        logging.info('nodes deleted')
        # nodes_to_del_id = 

        # dumper(['old nodes deleted, add new links'])

        for tpl in tpls_to_add:
            n0,n1 = tpl[0], tpl[1]
            self.g.add_edge(n0, n1)

        self.g.remove_edges_from(tpls_to_del)
        
        logging.info('graph modifications done')
                
        self.adj = np.array([(self.vd[e[0]], self.vd[e[1]]) for e in self.g.edges()])
        self.node_names = [i for i in self.g.nodes()]

        self.set_node_positions(nodes_to_add)

            

    def set_node_positions(self, nodes_to_add):
        """set positions of new nodes"""
        
        for n in nodes_to_add:
            logging.info(['set new position of ', n])


            # node_rect = fm.boundingRect(n)
            # node_sz = (node_rect.width() + self.wd_pad*2, node_rect.height())
            
            v_prnts = list(set(self.g.predecessors(n)) - set(nodes_to_add))

            logging.info(['node prnts: ', v_prnts])
            if len(v_prnts) > 0:
                self.g.nodes[n]['x'] = self.g.nodes[v_prnts[0]]['x']
                self.g.nodes[n]['y'] = self.g.nodes[v_prnts[0]]['y']
            else:
                # if all is new: random assignment:
                
                self.g.nodes[n]['x'] = choices(range(100, self.width - 100))[0]
                self.g.nodes[n]['y'] = choices(range(100, self.height - 100))[0]
                

        
        logging.info('node positions adjusted')
        
        self.width = self.size().width()
        self.height = self.size().height()
        # print(self.width, self.height)

        self.t = self.init_t

    def recalculate_layout(self):
        """overall command to manage layout re-calculations"""
        self.base_pos_ar = np.array([(self.g.nodes[i]['x'],self.g.nodes[i]['y']) for i in self.g.nodes])
        
        if self.layout_type == 'force': 
            self.recalc_layout_force()
            
        if self.layout_type == 'dot':
            self.recalc_layout_dot()
            
        logging.info(['layout: ', self.layout_type])

    def recalc_layout_force(self):
        """calculate new change_array"""
        logging.info('force recalculating starting')
        
        # get node array
        # self.base_pos_ar = np.array([(self.g.nodes[i]['x'],self.g.nodes[i]['y']) for i in self.g.nodes])
        # base_pos_ar = np.array([(g.nodes[i]['x'],g.nodes[i]['y']) for i in g.nodes])
        
        pos_nds = np.copy(self.base_pos_ar)
        # pos_nds = pos_nds.astype('float64')
        pos_nds = pos_nds.astype('float32')

        A = nx.to_numpy_array(self.g)
        At = A.T
        A = A + At
        np.clip(A, 0, 1, out = A)
        # A = A.astype('float')
        
        # get corner points pos
        
        sqs = []
        for n in self.g.nodes():
            logging.info(['node', n])
            sqx = rect_points([self.g.nodes[n]['x'], self.g.nodes[n]['y'], 
                                    self.g.nodes[n]['width'], self.g.nodes[n]['height']])
            
            sqs.append(sqx)

        pos = np.concatenate(sqs)


        pos = pos.astype('float32')

        row_order = get_reorder_order_sliced(len(self.g.nodes))

        nbr_nds = A.shape[0]
        nbr_pts = pos.shape[0]

        # self.dim_ar = np.array([[self.g.nodes[i]['width'], self.g.nodes[i]['height']] for i in self.g.nodes])
        dim_ar2 = self.dim_ar.astype('float32')


        # pythran_res = pythran_itrtr_cbn(pos, pos_nds, A, row_order, dim_ar2, self.t, self.def_itr,
        #                         self.rep_nd_brd_start, self.k, self.height*1.0, self.width*1.0, grav_multiplier)

        # pos_nds = pythran_res[0]
        # ctr = pythran_res[2]

        
        # construct objects for seeing which nodes are edge label nodes
        elbl_pos_list = []
        elbl_cnct_nds = []
        c = 0
        for v in self.g.nodes:
            if self.g.nodes[v]['nd_tp'] == 'lbl':
                # g.nodes[v]['e_lbl'] = 1
                logging.info(["v: ", v, ", c: ", c])
                elbl_pos_list.append(c)
                cnct_nodes = list(self.g.predecessors(v)) + list(self.g.successors(v))
                logging.info(["connected nodes: ", cnct_nodes])
                elbl_cnct_nds.append([self.vd[cnct_nodes[0]], self.vd[cnct_nodes[1]]])
            c +=1
            
        elbl_pos_list = np.array(elbl_pos_list)
        elbl_cnct_nds = np.array(elbl_cnct_nds)
        
        logging.info(["elbl_pos_list:\n", elbl_pos_list])
        logging.info(["elbl_cnct_nds:\n", elbl_cnct_nds])

        t1 = time()
        ctr = 0

        grav_multiplier = 5.0

        logging.info(['pos_nds:\n', pos_nds])
        pos_nds = frucht(pos_nds, dim_ar2, self.k*1.0, A, self.width*1.0, self.height*1.0, self.t,
                         500, self.def_itr, self.rep_nd_brd_start,
                         elbl_pos_list, elbl_cnct_nds, 1.0
                         )
        logging.info(['pos_nds:\n', pos_nds])
        ctr = 0

        t2 = time()
        logging.info('calculated layout in ' + str(round(t2-t1,4)) + ' seconds with ' + str(ctr) + ' iterations')

        # base_pos_ar = np.array([(g.nodes[i]['x'],g.nodes[i]['y']) for i in g.nodes])

        # self.goal_vp = sfdp_layout(self.g, K=0.5, pos=self.pos_vp, **set_dict)
        # self.goal_vp = fruchterman_reingold_layout(self.g, pos = self.pos_vp)

        self.chng_ar = (pos_nds - self.base_pos_ar)/self.step

        # re-assign back to graph, just do once at end
        for i in zip(self.g.nodes, pos_nds):
            self.g.nodes[i[0]]['x'] = i[1][0]
            self.g.nodes[i[0]]['y'] = i[1][1]

        # print("base_pos_ar: ", self.base_pos_ar)
        logging.info('force recalculating done')


    def recalc_layout_dot(self):
        """recalculate node positions with dot layout"""
        logging.info('dot recalculating starting')
        t1 = time()
        # think i have to set it here as well, is defined in force calcs
        # maybe move to get_node_text_dimensions? 
        # self.dim_ar = np.array([[self.g.nodes[i]['width'], self.g.nodes[i]['height']] for i in self.g.nodes])

        # get node array
        self.base_pos_ar = np.array([(self.g.nodes[i]['x'],self.g.nodes[i]['y']) for i in self.g.nodes])
        dg = Digraph()
        
        dg.graph_attr = {'rankdir': 'BT', 'dpi': '72'}
        
        if self.use_edge_labels != True:
            dg.edges([i for i in self.g.edges])
        
        for n in self.g.nodes:
            # dg.node(i) = {'width': self.g.nodes[i]['width']/96, 'height': self.g.nodes[i]['height']/96}
            if self.g.nodes[n]['nd_tp'] == 'nd':
                dg.node(n, width = str(self.g.nodes[n]['width']/72), height = str(self.g.nodes[n]['height']/72))

                
        # assumes now that every edge has a label over which the out/in-going nodes can be retrieved
        if self.use_edge_labels == True:
            dg_edges = []

            for n in self.g.nodes:
                if self.g.nodes[n]['nd_tp'] == 'lbl':
                    node_out = list(self.g.predecessors(n))[0]
                    node_in = list(self.g.successors(n))[0]
                    dg.edge(node_out, node_in, label = self.g.nodes[n]['title'])
        
            logging.info(['graph:\n', str(dg)])
            
            # logging.info(['graph:\n', dg])
        
        dg_gv_piped = dg.pipe(format = 'json')
        dg_gv_parsed = json.loads(dg_gv_piped)


        # loop over edges? have to get get info somehow
        if self.use_edge_labels == True:

            for e in dg_gv_parsed['edges']:
                lbl_text = e['label']
                lbl_x, lbl_y = e['lp'].split(',')
                head_id = e['tail']
                tail_id = e['head']
                head_name = dg_gv_parsed['objects'][head_id]['name']
                tail_name = dg_gv_parsed['objects'][tail_id]['name']
                elbl_nd_name = elbl_nd_name = "lbl_" + head_name + "_" + tail_name + "_" + lbl_text
                self.g.nodes[elbl_nd_name]['x'] = float(lbl_x)
                # self.g.nodes[elbl_nd_name]['x'] = float(lbl_x) + self.g.nodes[elbl_nd_name]['width']/2
                self.g.nodes[elbl_nd_name]['y'] = float(lbl_y)

                
        for i in dg_gv_parsed['objects']:
            posx, posy = i['pos'].split(',')
            self.g.nodes[i['name']]['x'] = float(posx) + 20 # tf is this +20 doing here
            self.g.nodes[i['name']]['y'] = float(posy)

        pos_nds = np.array([(self.g.nodes[i]['x'],self.g.nodes[i]['y']) for i in self.g.nodes])
        logging.info(['pos_nds:\n', pos_nds])

        self.chng_ar = (pos_nds - self.base_pos_ar)/self.step
        
        t2 = time()
        
        logging.info('dot recalc done in ' + str(round(t2-t1)) + ' seconds')

    def dot_node_text(self, node_title, node_text_lines):
        """generate the label for dot export"""

        table_header = """<<table color='black' border = "0" cellborder ="0" cellspacing="-6" align = "right">"""
        table_node_title = """<tr><td align="left"><FONT POINT-SIZE="22">""" + node_title + "</FONT></td></tr>"

        table_lines = ["""<tr><td align="left">""" + i + "</td></tr>" for i in node_text_lines]
        table_end = "</table>>"

        node_text_string = table_header + table_node_title + "".join(table_lines) + table_end
        return node_text_string


    def export_graph(self, export_type, export_file):
        """export graph to different formats"""
        # if export_type == 'graphml':
        #     nx.write_graphml(self.g, export_file)
            
        if export_type == 'dot':
            dg = Digraph()
            dg.edges([i for i in self.g.edges])
        
            for i in self.g.nodes:
                node_text_lines_raw = self.node_texts_raw[i]
                node_text_lines =  [i for i in node_text_lines_raw.split('\n') if len(i) > 0]
                node_label = self.dot_node_text(i, node_text_lines)
                dg.node(i, label = node_label)
            dg.save(export_file)
            
            
        # export_file = 'test.svg'
        
        if export_type == 'svg':
            logging.info('export as svg')
            logging.info(['file', export_file])
            logging.info(['width: ', self.width])
            logging.info(['height: ', self.height])

            svg = QtSvg.QSvgGenerator()
            svg.setFileName(export_file)
            svg.setSize(QSize(int(self.width*0.8), int(self.height*0.8))) 
            # svg.setSize(QSize(self.width, self.height))
            
            # svg.PdmDpiX = 8
            # svg.PdmDpiY = 8
            
            # svg.setSize(QSize(600,700))
            # svg.setViewBox(QRect(0, 0, self.width, self.height))

            qp_svg = QPainter(svg)
            qp_svg.setRenderHint(QPainter.Antialiasing, True) 
            
            self.draw_edges(qp_svg)
            # have to add arbitrary scale factor
            self.draw_texts(qp_svg, 1.333)
            self.draw_rects(qp_svg)
            # rather split up paintEvent into more funtions
            
            qp_svg.end()
            


    def timer_func(self):

        self.qt_coords = self.base_pos_ar + self.chng_ar * self.ctr
        # # debugging
        # if math.isnan(self.qt_coords[0][0]):
        #     print('LUL')
        #     print(self.chng_ar)

        self.update()

        if self.ctr == self.step:
            
            self.base_pos_ar = self.qt_coords

            self.ctr = 0
            self.paint_timer.stop()


        self.ctr += 1

    
    
    def draw_arrow(self, qp, e):
        """draw arrow from p1 to rad units before p2"""
        # get arrow angle, counterclockwise from center -> east line

        p1x, p1y = e[0][0], e[0][1]
        p2x, p2y = e[1][0], e[1][1]
        p1_wd, p1_ht = e[2][0], e[2][1]
        p2_wd, p2_ht = e[3][0], e[3][1]
        # source = e[4][0]
        # target = e[4][1]

        # print('source: ', source, 'target: ', target)

        angle = math.atan2((p2x - p1x), (p2y - p1y))
        angle_rev = math.atan2((p1x - p2x), (p1y - p2y))
            
        ar_start_pt_d = get_edge_point_delta(p1_wd, p1_ht, angle)
            
            
        start_px = p1x + ar_start_pt_d[0]
        start_py = p1y + ar_start_pt_d[1]

        ar_goal_d = get_edge_point_delta(p2_wd, p2_ht, angle_rev)
        arw_goal_x = p2x + ar_goal_d[0]
        arw_goal_y =  p2y + ar_goal_d[1]

        # arrow stuff: +/- 30 deg
        
        ar1 = angle + math.radians(25)
        ar2 = angle - math.radians(25)

        arw_len = 10

        # print('angle: ', angle, 'ar1: ', ar1, '; ar2: ', ar2)

        # need to focus on vector from p2 to p1
        ar1_x = arw_goal_x - arw_len * math.sin(ar1)
        ar1_y = arw_goal_y - arw_len * math.cos(ar1)
        
        ar2_x = arw_goal_x - arw_len * math.sin(ar2)
        ar2_y = arw_goal_y - arw_len * math.cos(ar2)
        
        
        qp.drawLine(start_px, start_py, arw_goal_x, arw_goal_y)
        

        if self.draw_arrow_toggle == True and self.g.nodes[e[4][1]]['nd_tp'] == 'nd':
            qp.drawLine(ar1_x, ar1_y, arw_goal_x, arw_goal_y)
            qp.drawLine(ar2_x, ar2_y, arw_goal_x, arw_goal_y)

    def draw_edges(self, qp):
        """draw the edges"""
        
        edges = []
        for i in self.adj:
            edges.append((self.qt_coords[i[0]], self.qt_coords[i[1]], 
                          self.dim_ar[i[0]], self.dim_ar[i[1]], 
                          (self.node_names[i[0]], self.node_names[i[1]])))

        qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        [self.draw_arrow(qp, e) for e in edges]


    def draw_texts(self, qp, scl):
        """draw the node texts, add scale factor scl (around 1.333) for svg export """
        qp.setPen(QColor(168, 34, 2))

        # draw node titles and text

        # ##logging.info(['vflush: ', self.title_vflush])

        for t in zip(self.qt_coords, self.dim_ar, self.node_names):

            if self.g.nodes[t[2]]['nd_tp'] == 'nd':
                qp.setFont(QFont('Arial', self.font_size * scl))
                ypos = (t[0][1]-t[1][1]/2) + self.title_vflush/1.3333


            if self.g.nodes[t[2]]['nd_tp'] == 'lbl':
                qp.setFont(QFont('Arial', self.node_text_size * scl))

                # not clear if good solution
                ypos = (t[0][1]-t[1][1]/2) + self.node_text_vflush*0.75
                
            xpos = t[0][0]-t[1][0]/2+ self.wd_pad
            
            # qp.drawText(xpos, ypos, t[2])
            qp.drawText(xpos, ypos, self.g.nodes[t[2]]['title'])
            
            qp.setFont(QFont('Arial', self.node_text_size * scl))
            
            # node_text_lines_raw = self.node_texts_raw[t[2]]
            # node_text_lines =  [i for i in node_text_lines_raw.split('\n') if len(i) > 0]
            c = 1
            # for t2 in node_text_lines:
            # for t2 in self.node_texts_proc[t[2]]:
            
            for t2 in self.g.nodes[t[2]]['node_text']:
            
                qp.drawText(xpos, ypos + self.node_text_vflush*c, t2)
                c+=1
            

        
    def draw_rects(self, qp):
        """draw the rectangles of nodes"""
        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))

        

        for i in zip(self.qt_coords, self.dim_ar, self.node_names):
            if self.g.nodes[i[2]]['nd_tp'] == 'nd':
                if self.cur_node == i[2]:
                    qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
                    qp.drawRect(i[0][0]-i[1][0]/2, i[0][1]- i[1][1]/2, i[1][0], i[1][1])
                    qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))
                else:
                    qp.drawRect(i[0][0]-i[1][0]/2, i[0][1]- i[1][1]/2, i[1][0], i[1][1])


        self.width = self.size().width()
        self.height = self.size().height()


    def paintEvent(self, event):
        """actual paint event, now heavily functionalized"""
        t1 = time()

        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)

        self.draw_edges(qp)


        self.draw_texts(qp, 1.0)

        self.draw_rects(qp)

        t2 = time()
        
        logging.info('painting took ' + str(round(t2-t1,4)) + ' seconds')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-v', default=False, action='store_true')
    parser.add_argument('con_type')
    parser.add_argument('layout_type')
    
    args = parser.parse_args()
    con_type = args.con_type
    layout_type = args.layout_type
    
    if con_type == 'zmq':
        import zmq

        class ZeroMQ_Listener(QtCore.QObject):

            message = QtCore.pyqtSignal(str)

            def __init__(self):

                QtCore.QObject.__init__(self)

                # Socket to talk to server
                context = zmq.Context()
                self.socket = context.socket(zmq.SUB)

                self.socket.connect ("tcp://localhost:5556")
                logging.info("connected to server")


                self.socket.setsockopt(zmq.SUBSCRIBE, b'')

                self.running = True

            def loop(self):
                while self.running:
                    string = self.socket.recv()
                    self.message.emit(str(string, 'utf-8'))


                    # do the update stuff here? 
                    # that would be so weird

        
    if con_type == 'dbus':
        from PyQt5 import QtDBus

        # ---------- dbus connection start -------

        class QDBusServer(QObject):

            def __init__(self):
                QObject.__init__(self)
                self.dbusAdaptor = QDBusServerAdapter(self)


        class QDBusServerAdapter(QtDBus.QDBusAbstractAdaptor):
            message_base = QtCore.pyqtSignal(str)

            QtCore.Q_CLASSINFO("D-Bus Interface", "com.qtpad.dbus")
            QtCore.Q_CLASSINFO("D-Bus Introspection",
            '  <interface name="com.qtpad.dbus">\n'
            '    <property name="name" type="s" access="read"/>\n'
            '    <method name="echo">\n'
            '      <arg direction="in" type="s" name="phrase"/>\n'
            '    </method>\n'
            '  </interface>\n')


            def __init__(self, parent):
                super().__init__(parent)

            @pyqtSlot(str, result=str)
            def echo(self, phrase):
                self.message_base.emit(phrase)
                # print("phrase: " + phrase + " received in Adaptor object")
        # ---------- dbus connection end -------


    if args.v == True:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)


    # con_type = 'dbus'
    # layout_type = 'force'

    mw = obvz_window(con_type, layout_type)
    sys.exit(app.exec_())
    


