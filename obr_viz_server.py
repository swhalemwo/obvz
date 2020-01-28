import time
import zmq

from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5.QtCore import QTimer, Qt, QObject
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont
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

app = QApplication(sys.argv)



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

        self.top= 150
        self.left= 150
        self.width = 500
        self.height = 500
        self.InitWindow()
        self.x = 5

        self.W = self.L = 800
        self.area = self.W*self.L

        self.t = 8
        self.dt = self.t/(100)
        self.k = 30
        self.step = 20 # how many steps realignment takes

        self.adj = []
        self.node_names = []
        self.link_str = ""
        self.links = []
        self.vd = {}
        self.vdr = {}
        
        self.qt_coords = np.array([])

        self.g = nx.Graph()

        self.thread = QtCore.QThread()
        self.zeromq_listener = ZeroMQ_Listener()
        self.zeromq_listener.moveToThread(self.thread)
        self.thread.started.connect(self.zeromq_listener.loop)
        self.ctr = 0
        self.paint_timer = QtCore.QTimer(self, timeout=self.timer_func, interval=100)
        
        self.zeromq_listener.message.connect(self.signal_received)

        QtCore.QTimer.singleShot(0, self.thread.start)
        self.g = nx.Graph()


    def InitWindow(self):
        # self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    
    def signal_received(self, new_link_str):
        print('signal receiving')

        # main parsing/updates has to happen here
        self.update_graph(new_link_str)

        self.paint_timer.start()

        print(message)

    def update_graph(self, new_link_str):
        """set new links and nodes"""

        new_links = new_link_str.split(";")
        new_link_tpls = [i.split(" -- ") for i in new_links]

        links_to_add = list(set(new_links) - set(self.links))
        # links_to_add = list(set(new_links) - set(links))
        links_to_del = list(set(self.links) - set(new_links))
        links_to_del = list(set(self.links) - set(new_links))

        # setting new stuff
        self.links = new_links

        new_nodes = []
        for tpl in new_link_tpls:
            new_nodes.append(tpl[0])
            new_nodes.append(tpl[1])

        new_nodes_unique = list(set(new_nodes))
        
        nodes_to_del = list(set(self.node_names) - set(new_nodes_unique))
        nodes_to_add = list(set(new_nodes_unique) - set(self.node_names))

        
        print("nodes_to_add: ", nodes_to_add)
        print("nodes_to_del: ", nodes_to_del)
        print("links_to_add: ", links_to_add)
        print("links_to_del: ", links_to_del)

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

        # have to reindex after deletion

        self.vd = {}
        self.vdr = {}
        c = 0
        for i in self.g.nodes():
            self.vd[i] = c
            self.vdr[c] = i
            c += 1
        
        # vd = {}
        # vdr = {}
        # c = 0
        # for i in g.nodes():
        #     vd[i] = c
        #     vdr[c] = i
        #     c += 1
        
        print('nodes deleted')
        # nodes_to_del_id = 

        # dumper(['old nodes deleted, add new links'])

        for l in links_to_add:
            tpl = l.split(" -- ")
            n0,n1 = tpl[0], tpl[1]
            self.g.add_edge(n0, n1)

        # for l in links_to_add:
        #     tpl = l.split(" -- ")
        #     n0,n1 = tpl[0], tpl[1]
        #     g.add_edge(n0, n1)

        # dumper(['new links added, delete old links'])

        for l in links_to_del:
            tpl = l.split(" -- ")
            n0, n1 = tpl[0], tpl[1]

            print([list(self.vd.keys())])
            # only remove edge when neither of nodes removed
            if n0 in self.vd.keys() and n1 in self.vd.keys():
                # self.g.remove_edge(self.g.edge(self.vd[n0], self.vd[n1]))
                self.g.remove_edge(n0, n1)

        print('graph modifications done')

        # set positions of new nodes to parent nodes
        # for n in nodes_to_add:
        #     v = self.g.vertex(self.vd[n])
        #     v_prnt = list(v.all_neighbors())[0]
        #     self.pos_vp[v] = self.pos_vp[v_prnt]
            
        print('setting attributes')
        # if all is new: random assignment:
        if len(nodes_to_add) == len(self.g.nodes):
            for v in self.g.nodes:
                self.g.nodes[v]['x'] = choices(range(100, 700))[0]
                self.g.nodes[v]['y'] = choices(range(100, 700))[0]
                
                self.g.nodes[v]['width'] = choices(range(10,80))[0]
                self.g.nodes[v]['height'] = choices(range(10,80))[0]


        else:
            for n in nodes_to_add:
                v_prnt = list(self.g.neighbors(n))[0]
                self.g.nodes[n]['x'] = self.g.nodes[n]['x']
                self.g.nodes[n]['y'] = self.g.nodes[n]['y']


            
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
        
        self.recalculate_layout()
        print('to here2')
        print(self.g.nodes)

    def recalculate_layout(self):
        """calculate new change_array, set rwr_c counter"""
        print('recalculating starting')
        
        # self.base_pos_ar = self.pos_vp.get_2d_array((0,1)).T
        # get node array
        self.base_pos_ar = np.array([(self.g.nodes[i]['x'],self.g.nodes[i]['y']) for i in self.g.nodes])
        # base_pos_ar = np.array([(g.nodes[i]['x'],g.nodes[i]['y']) for i in g.nodes])
        
        pos_nds = np.copy(self.base_pos_ar)
        pos_nds = pos_nds.astype('float64')

        A = nx.to_numpy_array(self.g)
        
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
        
        dim_ar = np.array([[self.g.nodes[i]['width'], self.g.nodes[i]['height']] for i in self.g.nodes])
        t1 = time()
        ctr = 0

        while True:
            # avoid node overlap at end
            if self.t < self.dt*30:
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
            displacement[:,0] += (self.k*10)**2/(pos_nds[:,0] - dim_ar[:,0]/2)**2
            displacement[:,1] += (self.k*10)**2/(pos_nds[:,1] - dim_ar[:,1]/2)**2
            displacement[:,0] -= (self.k*10)**2/(self.width - (pos_nds[:,0] + dim_ar[:,0]/2))**2
            displacement[:,1] -= (self.k*10)**2/(self.height - (pos_nds[:,1] + dim_ar[:,1]/2))**2

            length = np.linalg.norm(displacement, axis=-1)
            length = np.where(length < 0.01, 0.1, length)
            delta_pos = np.einsum('ij,i->ij', displacement, self.t / length)

            # update node positions
            pos_nds += delta_pos
            # scale delta_pos to corner poitns
            delta_pos_xtnd = np.reshape(np.hstack([delta_pos]*4), (nbr_pts, 2))
            pos += delta_pos_xtnd

            
            if self.t > self.dt*30:
                self.t -= self.dt

            else: 
                # if self.t < self.dt*40:
                print(np.sum(both_ovlp))
                if np.sum(both_ovlp) == nbr_nds:
                    self.t -= self.dt

            ctr += 1
            print('temperature: ', self.t)
            if self.t < 0: 
                break

        t2 = time()
        print('calculated layout in ' + str(t2-t1) + 'seconds with ' + str(ctr) + ' iterations')

        # base_pos_ar = np.array([(g.nodes[i]['x'],g.nodes[i]['y']) for i in g.nodes])
        
        # self.goal_vp = sfdp_layout(self.g, K=0.5, pos=self.pos_vp, **set_dict)
        # self.goal_vp = fruchterman_reingold_layout(self.g, pos = self.pos_vp)

        
        # goal_ar = self.goal_vp.get_2d_array([0,1]).T

        self.chng_ar = (pos_nds - self.base_pos_ar)/self.step

        
        # self.rwr_c = self.step
        print("base_pos_ar: ", self.base_pos_ar)
        print("goal_ar: ", pos)
        print("chng_ar: ", self.chng_ar)
        print('recalculating done')



    def timer_func(self):
        print('lol')
        self.qt_coords = self.base_pos_ar + self.chng_ar * self.ctr
    
        print(self.ctr)

        self.ctr += 1
        self.update()
        if self.ctr == self.step:
            self.paint_timer.stop()
            self.ctr = 0

    def draw_arrow(self, qp, p1x, p1y, p2x, p2y, node_width):
        """draw arrow from p1 to rad units before p2"""
        # get arrow angle, counterclockwise from center -> east line

        # dumper(['painting time'])
        angle = math.degrees(math.atan2((p1y-p2y), (p1x - p2x)))

        # calculate attach point
        arw_goal_x = p2x + node_width * math.cos(math.radians(angle))
        arw_goal_y = p2y + node_width * math.sin(math.radians(angle))

        # calculate start point: idk how trig works but does
        start_px = p1x - node_width * math.cos(math.radians(angle))
        start_py = p1y - node_width * math.sin(math.radians(angle))

        # arrow stuff: +/- 30 deg
        ar1 = angle + 25
        ar2 = angle - 25

        arw_len = 10

        # need to focus on vector from p2 to p1
        ar1_x = arw_goal_x + arw_len * math.cos(math.radians(ar1))
        ar1_y = arw_goal_y + arw_len * math.sin(math.radians(ar1))
                
        ar2_x = arw_goal_x + arw_len * math.cos(math.radians(ar2))
        ar2_y = arw_goal_y + arw_len * math.sin(math.radians(ar2))
        
        # qp.drawLine(p1x, p1y, p2x, p2y)
        # qp.drawLine(p1x, p1y, arw_goal_x, arw_goal_y)
        qp.drawLine(start_px, start_py, arw_goal_x, arw_goal_y)
        qp.drawLine(ar1_x, ar1_y, arw_goal_x, arw_goal_y)
        qp.drawLine(ar2_x, ar2_y, arw_goal_x, arw_goal_y)


    def paintEvent(self, event):
        # dumper(['start painting'])
        node_width = 10
        qp = QPainter(self)
        edges = [(self.qt_coords[i[0]], self.qt_coords[i[1]]) for i in self.adj]
        # dumper([str(i) for i in edges])

        qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))

        # [qp.drawLine(e[0][0], e[0][1], e[1][0], e[1][1]) for e in edges]
        [self.draw_arrow(qp, e[0][0], e[0][1], e[1][0], e[1][1], (node_width/2) + 5) for e in edges]
        
        qp.setPen(QColor(168, 34, 3))

        qp.setFont(QFont('Decorative', 10))
        [qp.drawText(t[0][0] + node_width, t[0][1], t[1]) for t in zip(self.qt_coords, self.node_names)]


        qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        # qp.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        
        print('painting nodes')

        qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))

        for i in zip(self.qt_coords, self.node_names):
                qp.drawRect(i[0][0]-(node_width/2), i[0][1]- (node_width/2), node_width, node_width)







if __name__ == "__main__":


    mw = ZeroMQ_Window()
    sys.exit(app.exec_())
    


# * test
# link_str = "ed -- norma -- secretly dating?;ed -- nadine -- officially together;ed -- james -- uncle;ed -- garage -- owns;norma -- shally -- work tgt bond over marrige problems;norma -- hank -- married;norma -- meals_on_wheels -- coorganized;norma -- cafe -- owns;norma -- ed -- secretly dating?;nadine -- garage -- works at;nadine -- ed -- officially together;log_lady -- laura -- saw_at_night;hank -- josie -- some relation;hank -- norma -- married;bernard -- jacque -- brothres;bernard -- coke -- transport;jacque -- leo -- coke trade partners?;jacque -- bernard -- brothres;josie -- hank -- some relation;josie -- ben_horne -- suspects of hurting her;josie -- cathrin -- suspects of hurting her;josie -- laura -- hires for english;josie -- sawmill -- ownes;josie -- truman -- date;dr_jacobi -- red_corvette_man -- claims to have seen;dr_jacobi -- coke -- also involed with?;dr_jacobi -- johnny -- treats;dr_jacobi -- necklace -- dug out?;dr_jacobi -- laura -- some fucked up relationship;leo -- bernard -- killed?;leo -- red_corvette_man -- is?;leo -- jacque -- coke trade partners?;leo -- coke -- sells;leo -- shally -- married;shally -- norma -- work tgt bond over marrige problems;shally -- cafe -- works_at;shally -- bobbie -- secretly dating;shally -- leo -- married;truman -- james -- hired_for_militia;truman -- ed -- hired_for_police_stuff;truman -- albert -- hits;truman -- josie -- date"
