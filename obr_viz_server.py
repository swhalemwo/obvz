import time 
import zmq

import json

from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5.QtCore import QTimer, Qt, QObject
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QFontMetrics
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.spatial import distance as dist

from time import sleep, time


import sys
import signal

import math

import numpy as np
from random import sample, choices

from collections import Counter

import networkx as nx

from PIL import ImageFont

from ovlp_func import pythran_itrtr


app = QApplication(sys.argv)

def get_edge_point_delta(wd, ht, angle):
    """get attach point of edge for node"""

    # print(angle)

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


def get_dim_ovlp(pos, row_order):
    """see if rectangles (indicated by 4 corner points) overlap, 
    has to be called with just 1D of points
    """
    
    nbr_nds = int(pos.shape[0]/4)
    nbr_pts = pos.shape[0]

    d = pos[:,np.newaxis] - pos[np.newaxis,:]
    d_rshp = np.reshape(d, (int((nbr_pts**2)/4), 4))
    d_rshp_rord = d_rshp[row_order]
    d_rord2 = np.reshape(d_rshp_rord, (nbr_nds, nbr_nds, 16))
    d_min = np.min(d_rord2, axis = 2)
    d_max = np.max(d_rord2, axis = 2)
    
    d_shrt = np.min(np.abs(d_rord2), axis = 2)
    
    d_ovlp = d_min * d_max
    
    d_ovlp[np.where(d_ovlp == 0)] = 1

    d_ovlp2 = (np.abs(d_ovlp)/d_ovlp)*(-1)
    np.clip(d_ovlp2, 0, 1, out = d_ovlp2)
    # return d_ovlp2, np.abs(d_min)
    return d_ovlp2, d_shrt

def get_point_mat_reorder_order(point_nbr):
    """gets reorder sequence to re-arrange rows in point dist matrix"""
    test_order = []
    # point_nbr = 20

    for i in range(point_nbr):
            for k in range(4):
                test_order.append(i+ (point_nbr*k))

    test_order = np.array(test_order)

    test_orders = []
    for i in range(point_nbr):
        test_orders.append((test_order + (point_nbr*4*i)).tolist())

    row_order_final = [i for sub_list in test_orders for i in sub_list]
    return row_order_final


# attractive force
def f_a(self, d,k):
    return d*d/k

# repulsive force
def f_r(self, d,k):
    return k*k/d


def rect_points(r):
    p1 = [r[0] + (r[2]/2), r[1] + (r[3]/2)]
    p2 = [r[0] + (r[2]/2), r[1] - (r[3]/2)]
    p3 = [r[0] - (r[2]/2), r[1] + (r[3]/2)]
    p4 = [r[0] - (r[2]/2), r[1] - (r[3]/2)]

    return np.array([p1, p2, p3, p4])



class ZeroMQ_Listener(QtCore.QObject):

    message = QtCore.pyqtSignal(str)
    
    def __init__(self):

        QtCore.QObject.__init__(self)
        
        # Socket to talk to server
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)

        print("Collecting updates from weather server")
        self.socket.connect ("tcp://localhost:5556")

        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        self.running = True

    def loop(self):
        while self.running:
            string = self.socket.recv()
            self.message.emit(str(string, 'utf-8'))

            
            # do the update stuff here? 
            # that would be so weird


class ZeroMQ_Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.top= 0
        self.left= 0
        self.InitWindow()

        # print(self.width, self.height)

        self.font_size = 13
        
        # self.area = self.width * self.height
        self.wd_pad = 8 # padding to add to sides of node for better drawing

        self.init_t = 12.0 # initial temperature: how much displacement possible per move, decreases
        self.def_itr = 100 # minimum number of iterations
        
        self.dt = self.init_t/(self.def_itr) # delta temperature, (int) sets number of iterations
        self.rep_nd_brd_start = 0.3 # relative size of end time frame in which node borders become repellant
        self.k = 30.0 # desired distance? 
        self.step = 6.0 # how many steps realignment takes
        self.update_interval = 40
        

        self.adj = [] # adjacency list? 
        self.node_names = []
        self.link_str = ""
        self.node_texts_raw = {}
        self.node_texts = {}
        self.links = []
        self.tpls = []
        self.vd = {}
        self.vdr = {}
        
        self.qt_coords = np.array([])
        self.dim_ar = np.array([])

        self.g = nx.DiGraph()

        self.thread = QtCore.QThread()
        self.zeromq_listener = ZeroMQ_Listener()
        self.zeromq_listener.moveToThread(self.thread)
        self.thread.started.connect(self.zeromq_listener.loop)
        self.ctr = 0
        self.paint_timer = QtCore.QTimer(self, timeout=self.timer_func, interval=self.update_interval)
        
        self.zeromq_listener.message.connect(self.signal_received)

        QtCore.QTimer.singleShot(0, self.thread.start)
        

    def InitWindow(self):
        # self.setWindowTitle(self.title)
        # self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    
    def redraw_layout(self, cmd):
        # assign new random positions
        if cmd == "hard":
            for n in self.g.nodes():
                
                self.g.nodes[n]['x'] = choices(range(int(self.g.nodes[n]['width']/2) + 10, self.width - (int(self.g.nodes[n]['width']/2) + 10)))[0]
                self.g.nodes[n]['y'] = choices(range(int(self.g.nodes[n]['height']/2) + 10, self.height - (int(self.g.nodes[n]['height']/2) + 10)))[0]

        # just apply forces to current layout
        # do i even need condition? not really i think, condition is based on what calls this function
        # if cmd == "soft":
            
        self.t = self.init_t
        self.recalculate_layout()
        self.paint_timer.start()


    def reset(self):
        self.adj = [] # adjacency list? 
        self.node_names = []
        self.link_str = ""
        self.node_texts = {}
        self.links = []
        self.tpls = []
        self.vd = {}
        self.vdr = {}
        self.node_texts_raw = {}
        
        self.qt_coords = np.array([])
        self.dim_ar = np.array([])

        self.g = nx.DiGraph()
        self.update()

    
    def signal_received(self, new_graph_str):
        print('signal receiving')
        # main parsing/updates has to happen here
        new_graph_dict = json.loads(new_graph_str)
    
        # only command is to redraw
        if list(new_graph_dict.keys())[0] == 'redraw':
            self.redraw_layout(new_graph_dict['redraw'])
            
        # update current node or graph
        # links and node texts as separate things
        else:
            self.cur_node = new_graph_dict['cur_node']
            
            if new_graph_dict['links'] == None:
                self.reset()

            update_me = 0
            # check if graph structure has changed
            if self.link_str != new_graph_dict['links'] and new_graph_dict['links'] != None:
                print('graph has changed')
                self.update_graph(new_graph_dict['links'])
                self.link_str = new_graph_dict['links']
                update_me = 1

            # check if node texts have been modified
            if self.node_texts_raw != new_graph_dict['node_texts'] and new_graph_dict['node_texts'] != None:
                print('node texts have changed')
                self.set_node_wd_ht(list(self.g.nodes()), new_graph_dict['node_texts'])
                self.node_texts_raw = new_graph_dict['node_texts']
                update_me = 1

            # start the layout calculations from here
            if update_me == 1:
                self.recalculate_layout()
                self.paint_timer.start()
            
                
            # no change: just make sure redrawing is done to take cur_node into account
            # if self.link_str == new_graph_dict['links'] and new_graph_dict['node_texts'] == self.node_texts:
            if update_me == 0:
                print('graph is same, just update current node')
                self.update()
                # print(new_link_str)

    # def update_graph(self, new_link_str):


    def get_node_text_dimensions(self, fm_nt, node_text):
        
        # n = 'bobbie'
        # node_text = new_graph_dict['node_texts'][n]
        # font = QFont("Arial", 10)
        # fm_nt = QFontMetrics(font) # font metric node text
        # ---------- test values end ----------
        
        # maybe implement some wrapping of long lines
        # but would have to determine where to wrap them, might be not straightforward with long lines

        node_text_lines = [i for i in node_text.split('\n') if len(i) > 0]
                
        node_rects = [fm_nt.boundingRect(i) for i in node_text_lines]
        widths = [i.width() for i in node_rects]
        heights = [i.height() for i in node_rects]

        return(widths, heights)


    def set_node_wd_ht(self, nodes_to_recalc_dims, node_text_dict):
        """set height and width attributes based on text properties"""
        # hm should it be so general that i don't have to run it every time? 
        # question is if i recalculate all node properties if graph changes
        # depends on how expensive it is
        # either way should avoid multiple instantiations of fm_nt
        # also need to split up position assignment and height/width calculation 
        
        print('setting attributes')

        font = ImageFont.truetype('Arial', self.font_size)
        
        # font = QFont("Arial", 12)
        font = QFont("Arial", self.font_size)
        fm = QFontMetrics(font)


        font = QFont("Arial", 10)
        fm_nt = QFontMetrics(font) # font metric node text


        for n in nodes_to_recalc_dims:
            node_rect = fm.boundingRect(n)
            nd_title_wd, nd_title_ht = node_rect.width(), node_rect.height()
            
            nt_dims = self.get_node_text_dimensions(fm_nt, node_text_dict[n])
            nt_dims[0].append(nd_title_wd)
            nt_dims[1].append(nd_title_ht)

            # node_sz = (node_rect.width() + self.wd_pad*2, node_rect.height())
            # self.g.nodes[n]['width'] = node_sz[0]
            # self.g.nodes[n]['height'] = node_sz[1]

            self.g.nodes[n]['width'] = max(nt_dims[0]) + self.wd_pad*2
            self.g.nodes[n]['height'] = sum(nt_dims[1])



    def update_graph(self, new_links):
        
        """set new links and nodes"""

        # need clear names: 
        # "new": refers to new graph as a whole (can exist before), 
        # "old": being in old graph, irrespective of being in new one
        # "add": existing in new, not in old
        # "del": in old, not in new

        # new_links = new_link_str.split(";")
        new_tpls = [(i.split(" -- ")[0], i.split(" -- ")[1]) for i in new_links]
        
        tpls_to_add = list(set(new_tpls) - set(self.tpls))
        tpls_to_del = list(set(self.tpls) - set(new_tpls))
        
        # not clear if links (one string) are that relevant, tuples seem to more convenient to work with tbh
        old_tpls = self.tpls
        self.tpls = new_tpls
        # self.links = new_links
        
        old_nodes = []
        for l in old_tpls:
            old_nodes.append(l[0])
            old_nodes.append(l[1])

        old_nodes = set(old_nodes)

        new_nodes = []
        for l in self.tpls:
            new_nodes.append(l[0])
            new_nodes.append(l[1])
        new_nodes = set(new_nodes)
        
        
        nodes_to_del = list(old_nodes - new_nodes)
        nodes_to_add = list(new_nodes - old_nodes)

        print('new tpls: ', new_tpls)
        print('old tpls: ', old_tpls)

        print("links_to_add: ", tpls_to_add)
        print("links_to_del: ", tpls_to_del)

        print('new nodes: ', new_nodes)
        print('old nodes: ', old_nodes)
        
        print("nodes_to_add: ", nodes_to_add)
        print("nodes_to_del: ", nodes_to_del)
        
        # first add nodes + index them
        # actually not clear if vd is needed much
        index_pos = len(self.g.nodes)

        for n in nodes_to_add:
            print('adding node')
            
            self.g.add_node(n)
            self.vd[n] = index_pos
            self.vdr[index_pos] = n
            index_pos +=1

        self.g.remove_nodes_from(nodes_to_del)
        print('nodes deleted')
        # have to reindex after deletion

        self.vd = {}
        self.vdr = {}
        c = 0
        for i in self.g.nodes():
            self.vd[i] = c
            self.vdr[c] = i
            c += 1
        
        print(self.vd)
        
        print('nodes deleted')
        # nodes_to_del_id = 

        # dumper(['old nodes deleted, add new links'])

        for tpl in tpls_to_add:
            n0,n1 = tpl[0], tpl[1]
            self.g.add_edge(n0, n1)

        self.g.remove_edges_from(tpls_to_del)
        # print('g edges after deleting: ', self.g.edges)
        # print('nbr nodes: ', len(self.g.nodes), "    nbr edges: ", len(self.g.edges))
        
        print('graph modifications done')
                
        self.adj = np.array([(self.vd[e[0]], self.vd[e[1]]) for e in self.g.edges()])
        self.node_names = [i for i in self.g.nodes()]

        self.set_node_positions(nodes_to_add)

            

    def set_node_positions(self, nodes_to_add):
        """set positions of new nodes"""
        
        for n in nodes_to_add:
            print('set new position of ', n)
            # print('neighbors: ', set(self.g.neighbors(n)))
            # print('nodes to add: ', nodes_to_add)


            # node_rect = fm.boundingRect(n)
            # node_sz = (node_rect.width() + self.wd_pad*2, node_rect.height())
            
            v_prnts = list(set(self.g.predecessors(n)) - set(nodes_to_add))
            # print('node: ', n)
            print('node prnts: ', v_prnts)
            if len(v_prnts) > 0:
                self.g.nodes[n]['x'] = self.g.nodes[v_prnts[0]]['x']
                self.g.nodes[n]['y'] = self.g.nodes[v_prnts[0]]['y']
            else:
                # if all is new: random assignment:
                
                self.g.nodes[n]['x'] = choices(range(100, self.width - 100))[0]
                self.g.nodes[n]['y'] = choices(range(100, self.height - 100))[0]
                
                # self.g.nodes[n]['x'] = choices(range(node_sz[0] + 10, self.width - (node_sz[0] + 10)))[0]
                # self.g.nodes[n]['y'] = choices(range(node_sz[1] + 10, self.height - (node_sz[1] + 10)))[0]

            # self.g.nodes[n]['width'] = node_sz[0]
            # self.g.nodes[n]['height'] = node_sz[1]

        
        print('node positions adjusted')
        
        self.width = self.size().width()
        self.height = self.size().height()
        # print(self.width, self.height)

        self.t = self.init_t
        # self.recalculate_layout()
        # print('to here2')
        # print(self.g.nodes)

    def recalculate_layout(self):
        """calculate new change_array, set rwr_c counter"""
        print('recalculating starting')
        
        # get node array
        self.base_pos_ar = np.array([(self.g.nodes[i]['x'],self.g.nodes[i]['y']) for i in self.g.nodes])
        # base_pos_ar = np.array([(g.nodes[i]['x'],g.nodes[i]['y']) for i in g.nodes])
        
        pos_nds = np.copy(self.base_pos_ar)
        pos_nds = pos_nds.astype('float64')

        A = nx.to_numpy_array(self.g)
        At = A.T
        A = A + At
        np.clip(A, 0, 1, out = A)
        
        # get corner points pos
        
        sqs = []
        for i in self.g.nodes():
            sqx = rect_points([self.g.nodes[i]['x'], self.g.nodes[i]['y'], 
                                    self.g.nodes[i]['width'], self.g.nodes[i]['height']])
            sqs.append(sqx)

        pos = np.concatenate(sqs)

        pos = pos.astype('float64')

        # get row_order
        row_order = get_point_mat_reorder_order(len(self.g.nodes))

        nbr_nds = A.shape[0]
        nbr_pts = pos.shape[0]

        self.dim_ar = np.array([[self.g.nodes[i]['width'], self.g.nodes[i]['height']] for i in self.g.nodes])
        dim_ar2 = self.dim_ar.astype('float64')
        t1 = time()
        ctr = 0

        grav_multiplier = 5.0

        pythran_res = pythran_itrtr(pos, pos_nds, A, row_order, dim_ar2, self.t, self.def_itr,
                                    self.rep_nd_brd_start, self.k, self.height*1.0, self.width*1.0, grav_multiplier)

        pos_nds = pythran_res[0]
        ctr = pythran_res[2]

        t2 = time()
        print('calculated layout in ' + str(t2-t1) + 'seconds with ' + str(ctr) + ' iterations')

        # base_pos_ar = np.array([(g.nodes[i]['x'],g.nodes[i]['y']) for i in g.nodes])

        # self.goal_vp = sfdp_layout(self.g, K=0.5, pos=self.pos_vp, **set_dict)
        # self.goal_vp = fruchterman_reingold_layout(self.g, pos = self.pos_vp)

        self.chng_ar = (pos_nds - self.base_pos_ar)/self.step

        # re-assign back to graph, just do once at end
        for i in zip(self.g.nodes, pos_nds):
            self.g.nodes[i[0]]['x'] = i[1][0]
            self.g.nodes[i[0]]['y'] = i[1][1]


        # self.rwr_c = self.step
        # print("base_pos_ar: ", self.base_pos_ar)
        # print("goal_ar: ", pos)
        # print("chng_ar: ", self.chng_ar)
        print('recalculating done')



    def timer_func(self):

        self.qt_coords = self.base_pos_ar + self.chng_ar * self.ctr
        # print(self.qt_coords)
        
        # some_node = list(self.g.nodes())[5]
        # print('node position in base_pos_ar: ', self.base_pos_ar[5])
        # print('change vector: ', self.chng_ar[5])
        # print('ctr: ', self.ctr)
        # print('some node: ', some_node)
        # print('x: ', self.g.nodes[some_node]['x'], 'y: ', self.g.nodes[some_node]['y'])
        # print("some nodes' position: ", self.qt_coords[5])
        
            # print(self.ctr)


        self.update()
        if self.ctr == self.step:
            
            self.base_pos_ar = self.qt_coords

            self.ctr = 0
            self.paint_timer.stop()

        self.ctr += 1

    
    

    # def draw_arrow(self, qp, p1x, p1y, p2x, p2y, node_width):
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
        qp.drawLine(ar1_x, ar1_y, arw_goal_x, arw_goal_y)
        qp.drawLine(ar2_x, ar2_y, arw_goal_x, arw_goal_y)


    def paintEvent(self, event):
        
        t1 = time()
        node_width = 0
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        # edges = [(self.qt_coords[i[0]], self.qt_coords[i[1]]) for i in self.adj]
        edges = [(self.qt_coords[i[0]], self.qt_coords[i[1]], self.dim_ar[i[0]], self.dim_ar[i[1]], (self.node_names[i[0]], self.node_names[i[1]]))
                  for i in self.adj]
        
        # print(edges)

        qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))

        # [qp.drawLine(e[0][0], e[0][1], e[1][0], e[1][1]) for e in edges]
        # [self.draw_arrow(qp, e[0][0], e[0][1], e[1][0], e[1][1], (node_width/2) + 0) for e in edges]
        [self.draw_arrow(qp, e) for e in edges]
        # print(self.node_names[edges[-1][0]], self.node_names[edges[-1][1]])

        qp.setPen(QColor(168, 34, 2))

        # draw node titles
        qp.setFont(QFont('Arial', self.font_size))
        for t in zip(self.qt_coords, self.dim_ar, self.node_names): 
            # qp.drawText(t[0][0]-t[1][0]/2+ self.wd_pad, t[0][1] + 5 , t[2])
            
            xpos = t[0][0]-t[1][0]/2+ self.wd_pad
            ypos = (t[0][1]-t[1][1]/2) + 15
            qp.drawText(xpos, ypos, t[2])
            
            qp.setFont(QFont('Arial', 10))
            
            node_text_lines_raw = self.node_texts_raw[t[2]]
            node_text_lines =  [i for i in node_text_lines_raw.split('\n') if len(i) > 0]
            c = 1
            for t2 in node_text_lines:
                qp.drawText(xpos, ypos + 15*c, t2)
                c+=1
            
            qp.setFont(QFont('Arial', self.font_size))
        
        
        qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        # qp.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        
        # print('painting nodes')
        # print(self.size())

        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))

        for i in zip(self.qt_coords, self.dim_ar, self.node_names):
            if self.cur_node == i[2]:
                qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
                qp.drawRect(i[0][0]-i[1][0]/2, i[0][1]- i[1][1]/2, i[1][0], i[1][1])
                qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))
            else:
                qp.drawRect(i[0][0]-i[1][0]/2, i[0][1]- i[1][1]/2, i[1][0], i[1][1])

        self.width = self.size().width()
        self.height = self.size().height()
        # print(self.width, self.height)
        t2 = time()
        
        print('painting took ', t2-t1, ' seconds')


if __name__ == "__main__":


    mw = ZeroMQ_Window()
    sys.exit(app.exec_())
    


# * scrap


new_graph_str = "{\"links\":[\"bobbie -- james -- hates\",\"bobbie -- coke -- buys?\",\"bobbie -- mike -- friends\",\"bobbie -- leo -- owes 10k\",\"bobbie -- shally -- secretly dating\",\"bobbie -- laura -- officially together\"],\"cur_node\":\"bobbie\",\"node_texts\":{\"bobbie\":\"\\n\\n\\nis kinda reckless\\nknows about leo beating shally\\n\\nmight have killed someone? mentioned in e1\\n\\nfreaks out at funeral\\n\",\"james\":\"\\n\\npretty sentimental for a biker? \\ndoesn't want to go to laura's funeral\\nbut still shows up?\\n\",\"coke\":\"\\n\\n\",\"mike\":\"\\n\\nviolent psycho? \\none arm\\n\",\"leo\":\"\\n\\n\\ntext i want to get visualized\\n\\nbloody clothes\\nfucking controlling psycho that beats shally LUL\\n\\n\",\"shally\":\"\\n\\nkinda stupid? has like 0 motivation except let's fuck? \\njokes about funeral like a garbage person\\ngets gun\\n\",\"laura\":\"\\n\\ndaed between 0 and 4am\\nbite mark, self-inflicted? \\nsleeps with 3 men\\n- jamie?\\n- bobbie?\\n- ???\\n\\ncocaine\\ntwine: bound twice\\n\\ntied up twice\\n\"}}"
