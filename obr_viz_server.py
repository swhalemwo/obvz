import time
import zmq

from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont
from sklearn.metrics.pairwise import euclidean_distances

from time import sleep
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

        self.adj = []
        self.node_names = []
        self.link_str = ""
        self.links = []
        self.g_id = {}

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
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    
    def signal_received(self, message):
        print('signal receiving')

        # main parsing/updates has to happen here
        

        self.paint_timer.start()

        print(message)

    def update_graph(self, new_link_str):
        """set new links and nodes"""

        new_links = new_link_str.split(";")
        new_link_tpls = [i.split(" -- ") for i in new_links]

        links_to_add = list(set(new_links) - set(self.links))
        # links_to_add = list(set(new_links) - set(links))
        links_to_del = list(set(self.links) - set(new_links))
        links_to_del = list(set(links) - set(new_links))

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
        # int(v) would return index, but nx doesn't have them
        # needs own index dict
        # adding: stuff gets added at back -> can just increase

        
        
        index_pos = len(g.nodes)
        vd = {}
        vdr = {}

        for n in nodes_to_add:
            print('adding node')
            
            # self.g.add_node(n)
            # self.vd[n] = index_pos
            # self.vdr[index_pos] = n
            # index_pos +=1

            g.add_node(n)
            vd[n] = index_pos
            vdr[index_pos] = n
            index_pos +=1

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
        
        print('node deleted')
        # nodes_to_del_id = 

        # dumper(['old nodes deleted, add new links'])

        for l in links_to_add:
            tpl = l.split(" -- ")
            n0,n1 = tpl[0], tpl[1]
            self.g.add_edge(n0, n1)

        for l in links_to_add:
            tpl = l.split(" -- ")
            n0,n1 = tpl[0], tpl[1]
            g.add_edge(n0, n1)

        # dumper(['new links added, delete old links'])

        for l in links_to_del:
            tpl = l.split(" -- ")
            n0 = tpl[0]
            n1 = tpl[1]
            dumper([list(self.vd.keys())])
            # only remove edge when neither of nodes removed
            if n0 in self.vd.keys() and n1 in self.vd.keys():
                self.g.remove_edge(self.g.edge(self.vd[n0], self.vd[n1]))

        print('graph modifications done')

        # set positions of new nodes to parent nodes
        for n in nodes_to_add:
            v = self.g.vertex(self.vd[n])
            v_prnt = list(v.all_neighbors())[0]
            self.pos_vp[v] = self.pos_vp[v_prnt]
        
        print('node positions adjusted')
        
        self.adj = np.array([(int(i.source()), int(i.target())) for i in self.g.edges()])
        self.node_names = [self.g_id[i] for i in self.g.vertices()]

        # dumper(['storage objects updated'])

        # dumper(["nbr_edges new: ", str(len([i for i in self.g.edges()]))])
        # dumper(['nodes_to_add'] + nodes_to_add)
        # seems to work
        
      
        self.recalculate_layout()
        dumper(['to here2'])




    def timer_func(self):
        print('lol')

        self.ctr += 1
        print(self.ctr)
        self.x +=3
        self.update()

        if self.ctr == 10:
            self.paint_timer.stop()
            self.ctr = 0

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
link_str = "ed -- norma -- secretly dating?;ed -- nadine -- officially together;ed -- james -- uncle;ed -- garage -- owns;norma -- shally -- work tgt bond over marrige problems;norma -- hank -- married;norma -- meals_on_wheels -- coorganized;norma -- cafe -- owns;norma -- ed -- secretly dating?;nadine -- garage -- works at;nadine -- ed -- officially together;log_lady -- laura -- saw_at_night;hank -- josie -- some relation;hank -- norma -- married;bernard -- jacque -- brothres;bernard -- coke -- transport;jacque -- leo -- coke trade partners?;jacque -- bernard -- brothres;josie -- hank -- some relation;josie -- ben_horne -- suspects of hurting her;josie -- cathrin -- suspects of hurting her;josie -- laura -- hires for english;josie -- sawmill -- ownes;josie -- truman -- date;dr_jacobi -- red_corvette_man -- claims to have seen;dr_jacobi -- coke -- also involed with?;dr_jacobi -- johnny -- treats;dr_jacobi -- necklace -- dug out?;dr_jacobi -- laura -- some fucked up relationship;leo -- bernard -- killed?;leo -- red_corvette_man -- is?;leo -- jacque -- coke trade partners?;leo -- coke -- sells;leo -- shally -- married;shally -- norma -- work tgt bond over marrige problems;shally -- cafe -- works_at;shally -- bobbie -- secretly dating;shally -- leo -- married;truman -- james -- hired_for_militia;truman -- ed -- hired_for_police_stuff;truman -- albert -- hits;truman -- josie -- date"
