import time
import zmq

from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5.QtCore import QTimer, Qt, QObject
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QFontMetrics
from sklearn.metrics.pairwise import euclidean_distances

from time import sleep, time
from scipy.spatial import distance as dist

import sys
import signal

import math

import numpy as np
from random import sample, choices

from collections import Counter

import networkx as nx

from PIL import ImageFont


app = QApplication(sys.argv)

def get_edge_point_delta(wd, ht, angle):
    """get attach point of edge for node"""
    rel_angle = abs(angle) % math.pi

    if angle >= 0 and angle < math.pi/2: sector = 1
    elif angle > math.pi/2: sector = 2
    elif angle < -math.pi/2: sector = 3
    elif angle < 0 and angle > -math.pi/2: sector = 4

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

        self.init_t = 12 # initial temperature: how much displacement possible per move, decreases
        self.def_iter = 100 # minimum number of iterations
        
        self.dt = self.init_t/(self.def_iter) # delta temperature, (int) sets number of iterations
        self.rep_nd_brd_start = 0.3 # relative size of end time frame in which node borders become repellant
        self.k = 30 # desired distance? 
        self.step = 30 # how many steps realignment takes

        self.adj = [] # adjacency list? 
        self.node_names = []
        self.link_str = ""
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
        self.paint_timer = QtCore.QTimer(self, timeout=self.timer_func, interval=30)
        
        self.zeromq_listener.message.connect(self.signal_received)

        QtCore.QTimer.singleShot(0, self.thread.start)
        

    def InitWindow(self):
        # self.setWindowTitle(self.title)
        # self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    
    def signal_received(self, new_link_str):
        print('signal receiving')

        # main parsing/updates has to happen here
        self.update_graph(new_link_str)

        self.paint_timer.start()

        # print(new_link_str)

    def update_graph(self, new_link_str):
        """set new links and nodes"""

        # need clear names: 
        # "new": refers to new graph as a whole (can exist before), 
        # "old": being in old graph, irrespective of being in new one
        # "add": existing in new, not in old
        # "del": in old, not in new

        new_links = new_link_str.split(";")
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
        # have to index differently now
        # int(v) would return index in gt, but nx doesn't have them
        # needs own index dict
        # adding: stuff gets added at back -> can just increase

        
        index_pos = len(self.g.nodes)
        # vd = {}
        # vdr = {}

        for n in nodes_to_add:
            print('adding node')
            
            self.g.add_node(n)
            self.vd[n] = index_pos
            self.vdr[index_pos] = n
            index_pos +=1

            # g.add_node(n)
            # vd[n] = index_pos
            # vdr[index_pos] = n
            # index_pos +=1

        # del_node_ids = [self.vd[i] for i in nodes_to_del]
        # self.g.remove_vertex(del_node_ids)
        
        
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
        
        print(list(self.vd.keys()))

        
        print('nodes deleted')
        # nodes_to_del_id = 

        # dumper(['old nodes deleted, add new links'])

        for tpl in tpls_to_add:
            n0,n1 = tpl[0], tpl[1]
            self.g.add_edge(n0, n1)

        # for l in links_to_add:
        #     tpl = l.split(" -- ")
        #     n0,n1 = tpl[0], tpl[1]
        #     g.add_edge(n0, n1)

        # dumper(['new links added, delete old links'])

        # print('nbr links to del: ' + str(len(links_to_del)))
        # print('nbr unq links to del: ' + str(len(list(set(links_to_del)))))

        # for l in set(links_to_del):
        #     tpl = l.split(" -- ")
        #     n0, n1 = tpl[0], tpl[1]
        #     print('delete edge between ' + n0 + ' and ' + n1)
        #     # print(self.g.nodes)
        #     # print(self.g.edges)
        #     # only remove edge when neither of nodes removed
        #     if n0 in self.vd.keys() and n1 in self.vd.keys():
        #         # self.g.remove_edge(self.g.edge(self.vd[n0], self.vd[n1]))
        #         self.g.remove_edge(n0, n1)

        # tpls_to_del = [(i.split(" -- ")[0], i.split(" -- ")[1]) for i in links_to_del]
        print('g edges before deleting: ', self.g.edges)
        print('nbr nodes: ', len(self.g.nodes), "    nbr edges: ", len(self.g.edges))
        self.g.remove_edges_from(tpls_to_del)
        print('g edges after deleting: ', self.g.edges)
        print('nbr nodes: ', len(self.g.nodes), "    nbr edges: ", len(self.g.edges))
        
        print('graph modifications done')
        print('directedness: ', self.g.is_directed())
        # set positions of new nodes to parent nodes
        # for n in nodes_to_add:
        #     v = self.g.vertex(self.vd[n])
        #     v_prnt = list(v.all_neighbors())[0]
        #     self.pos_vp[v] = self.pos_vp[v_prnt]
            
        print('setting attributes')
        # if all is new: random assignment:
        font = ImageFont.truetype('Arial', self.font_size)
        
        font = QFont("Arial", self.font_size)
        fm = QFontMetrics(font)
        
        # add coords for some new nodes

        for n in nodes_to_add:
            print('set new position')
            print('neighbors: ', set(self.g.neighbors(n)))
            print('nodes to add: ', nodes_to_add)

            node_rect = fm.boundingRect(n)
            node_sz = (node_rect.width() + self.wd_pad*2, node_rect.height())
            
            v_prnts = list(set(self.g.neighbors(n)) - set(nodes_to_add))
            print('node: ', n)
            print('pob prnts: ', v_prnts)
            if len(v_prnts) > 0:
                self.g.nodes[n]['x'] = self.g.nodes[v_prnts[0]]['x']
                self.g.nodes[n]['y'] = self.g.nodes[v_prnts[0]]['y']
            else:
                self.g.nodes[n]['x'] = choices(range(node_sz[0] + 10, self.width - (node_sz[0] + 10)))[0]
                self.g.nodes[n]['y'] = choices(range(node_sz[1] + 10, self.height - (node_sz[1] + 10)))[0]


            self.g.nodes[n]['width'] = node_sz[0]
            self.g.nodes[n]['height'] = node_sz[1]

        # for i in self.g.nodes:
        #     print(i, self.g.nodes[i])
        
        
        print('node positions adjusted')
        
        # self.adj = np.array([(int(i.source()), int(i.target())) for i in self.g.edges()])
        self.adj = np.array([(self.vd[e[0]], self.vd[e[1]]) for e in self.g.edges()])
        # adj = np.array([(vd[e[0]], vd[e[1]]) for e in g.edges()])

        
        # self.node_names = [self.g_id[i] for i in self.g.vertices()]
        self.node_names = [i for i in self.g.nodes()]
        
        # node_names = [i for i in g.nodes()]


        # dumper(['storage objects updated'])

        # dumper(["nbr_edges new: ", str(len([i for i in self.g.edges()]))])
        # dumper(['nodes_to_add'] + nodes_to_add)
        # seems to work
        
        self.width = self.size().width()
        self.height = self.size().height()
        # print(self.width, self.height)

        self.t = self.init_t
        self.recalculate_layout()
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

        # sqs = []
        # for i in g.nodes():
        #     sqx = rect_points([g.nodes[i]['x'], g.nodes[i]['y'], 
        #                        g.nodes[i]['width'], g.nodes[i]['height']])
        #     sqs.append(sqx)

            
        pos = np.concatenate(sqs)
        pos = pos.astype('float64')
        
        # get row_order
        row_order = get_point_mat_reorder_order(len(self.g.nodes))
        
        nbr_nds = A.shape[0]
        nbr_pts = pos.shape[0]
        
        self.dim_ar = np.array([[self.g.nodes[i]['width'], self.g.nodes[i]['height']] for i in self.g.nodes])
        t1 = time()
        ctr = 0

        while True:
            # avoid node overlap at end
            if self.t < self.dt * self.def_iter * self.rep_nd_brd_start :
                delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]
                
                x_ovlp, dx_min = get_dim_ovlp(pos[:,0], row_order)
                y_ovlp, dy_min = get_dim_ovlp(pos[:,1], row_order)
                
                both_ovlp = x_ovlp * y_ovlp
                x_ovlp2 = x_ovlp - both_ovlp
                y_ovlp2 = y_ovlp - both_ovlp

                none_ovlp = np.ones((nbr_nds, nbr_nds)) - both_ovlp - x_ovlp2 - y_ovlp2

                # also have to get the point distances for none_ovlp (then shortest)
                delta_pts = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dist_pts = np.linalg.norm(delta_pts, axis=-1)
                
                dist_rshp = np.reshape(dist_pts, (int((nbr_pts**2)/4), 4))
                dist_rshp_rord = dist_rshp[row_order]
                dist_rord = np.reshape(dist_rshp_rord, (nbr_nds, nbr_nds, 16))
                min_pt_dists = np.min(dist_rord, axis = 2)
                
                distance = (x_ovlp2 * dy_min) + (y_ovlp2 * dx_min) + (both_ovlp * 1) + (none_ovlp * min_pt_dists)
                np.clip(distance, 1, None, out = distance)

                
            # just consider nodes as points at first
            else: 
                delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]
                distance = np.linalg.norm(delta_nds, axis=-1)
                np.clip(distance, 1, None, out = distance)
            
            
            # update points
            displacement = np.einsum('ijk,ij->ik',
                                     delta_nds,
                                     (self.k * self.k / distance**2) - A * distance / self.k)

            # repellant borders
            displacement[:,0] += (self.k*10)**2/(pos_nds[:,0] - self.dim_ar[:,0]/2)**2
            displacement[:,1] += (self.k*10)**2/(pos_nds[:,1] - self.dim_ar[:,1]/2)**2
            displacement[:,0] -= (self.k*10)**2/(self.width - (pos_nds[:,0] + self.dim_ar[:,0]/2))**2
            displacement[:,1] -= (self.k*10)**2/(self.height - (pos_nds[:,1] + self.dim_ar[:,1]/2))**2

            length = np.linalg.norm(displacement, axis=-1)
            length = np.where(length < 0.01, 0.1, length)
            delta_pos = np.einsum('ij,i->ij', displacement, self.t / length)

            # update node positions
            pos_nds += delta_pos
            # scale delta_pos to corner poitns
            delta_pos_xtnd = np.reshape(np.hstack([delta_pos]*4), (nbr_pts, 2))
            pos += delta_pos_xtnd

            
            if self.t > self.dt * self.def_iter * self.rep_nd_brd_start:
                # self.dt*30:
                self.t -= self.dt

            else: 
                # if self.t < self.dt*40:
                # print(np.sum(both_ovlp))
                if np.sum(both_ovlp) == nbr_nds:
                    self.t -= self.dt

            ctr += 1
            # print('temperature: ', self.t)
            if self.t < 0: 
                break

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
    
        # print(self.ctr)


        self.update()
        if self.ctr == self.step:
            self.paint_timer.stop()
            self.ctr = 0

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

        # qp.setFont(QFont('Decorative', 10))
        qp.setFont(QFont('Arial', self.font_size))
        [qp.drawText(t[0][0]-t[1][0]/2+ self.wd_pad, t[0][1] + 5 , t[2]) for t in zip(self.qt_coords, self.dim_ar, self.node_names)]


        qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        # qp.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        
        # print('painting nodes')
        # print(self.size())

        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))

        for i in zip(self.qt_coords, self.dim_ar):
                qp.drawRect(i[0][0]-i[1][0]/2, i[0][1]- i[1][1]/2, i[1][0], i[1][1])

        self.width = self.size().width()
        self.height = self.size().height()
        # print(self.width, self.height)
        t2 = time()
        
        print('painting took ', t2-t1, ' seconds')


if __name__ == "__main__":


    mw = ZeroMQ_Window()
    sys.exit(app.exec_())
    


