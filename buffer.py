from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from core.buffer import Buffer

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont
import sys
import signal

# import inspect
import csv

# from graph_tool.all import *
from graph_tool import Graph
from graph_tool.draw import fruchterman_reingold_layout

import numpy as np
from random import sample, choices

from numpy.random import poisson

from collections import Counter

from math import atan2, degrees, sin, cos, radians

def dumper(list_of_strings):
    """help function because idk how else to debug eaf/pyqt stuff"""
    with open("/tmp/eaf.csv", "a") as fo:
        wr = csv.writer(fo)
        [wr.writerow([i]) for i in list_of_strings]


class ob_viz(QWidget):
    
    def __init__(self, bg_color):
        QWidget.__init__(self)
        
        self.background_color = bg_color
        self.c = 0
        
        # K = 0.5
        # how many iterations the realignment is supposed to take
        self.step = 15
        self.rwr_c = 0

        # dumper([qt_coords])
        # self.show()

        # with open("/tmp/eaf3.csv", "a") as fo:
        #     wr = csv.writer(fo)
        #     wr.writerow([self.c, "runs4"])
        # dumper([self.c, "runs4"])
               

        # self.node_names [g_id[i] for i in g.vertices()]

    def init2(self, emacs_var_dict):
        self.emacs_var_dict = emacs_var_dict
        
        self.link_str = self.emacs_var_dict['links']
        self.g = Graph()
        self.label_ep = self.g.new_edge_property("string")
        self.links = self.link_str.split(";")

        link_tpls = [i.split(" -- ") for i in self.links]
        dumper([str(i) for i in link_tpls])
        
        self.g_id = self.g.add_edge_list(link_tpls, hashed = True, string_vals = True, eprops = [self.label_ep])
        
        self.adj = np.array([(int(i.source()), int(i.target())) for i in self.g.edges()])
        self.node_names = [self.g_id[i] for i in self.g.vertices()]

        self.vd = {}
        for i in self.g.vertices():
            self.vd[self.g_id[i]] = int(i)

        # self.pos_vp = sfdp_layout(self.g, K=0.5)
        self.pos_vp = fruchterman_reingold_layout(self.g)
        self.base_pos_ar = self.pos_vp.get_2d_array((0,1)).T
        self.qt_coords = self.nolz_pos_ar(self.base_pos_ar)

        dumper([str(self.qt_coords)])
        
        
        # dumper([link_str])


    def update_graph(self, emacs_var_dict):
        """set new links and nodes"""
        new_link_str = emacs_var_dict['links']
        new_links = new_link_str.split(";")
        new_link_tpls = [i.split(" -- ") for i in new_links]

        links_to_add = list(set(new_links) - set(self.links))
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

        
        dumper(["nodes_to_add: ", nodes_to_add,
                "nodes_to_del: ", nodes_to_del,
                "links_to_add: ", links_to_add,
                "links_to_del: ", links_to_del])

        # first add nodes + index them, but not there yet (first links)

        for n in nodes_to_add:
            dumper(['adding node'])
            v = self.g.add_vertex()
            # how to new nodes pos to parents? separate loop afterwards
            self.vd[n] = int(v)
            self.g_id[v] = n

        del_node_ids = [self.vd[i] for i in nodes_to_del]
        self.g.remove_vertex(del_node_ids)

        # have to reindex after deletion
        self.vd = {}
        for i in self.g.vertices():
            self.vd[self.g_id[i]] = int(i)
            
        dumper(['node deleted'])
        # nodes_to_del_id = 

        # dumper(['old nodes deleted, add new links'])

        for l in links_to_add:
            tpl = l.split(" -- ")
            n0,n1 = tpl[0], tpl[1]
            self.g.add_edge(self.vd[n0], self.vd[n1])
        
        # dumper(['new links added, delete old links'])

        for l in links_to_del:
            tpl = l.split(" -- ")
            n0 = tpl[0]
            n1 = tpl[1]
            dumper([list(self.vd.keys())])
            # only remove edge when neither of nodes removed
            if n0 in self.vd.keys() and n1 in self.vd.keys():
                self.g.remove_edge(self.g.edge(self.vd[n0], self.vd[n1]))

        # dumper(['graph modifications done'])

        # set positions of new nodes to parent nodes
        for n in nodes_to_add:
            v = self.g.vertex(self.vd[n])
            v_prnt = list(v.all_neighbors())[0]
            self.pos_vp[v] = self.pos_vp[v_prnt]
        
        # dumper(['node positions adjusted'])
        
        self.adj = np.array([(int(i.source()), int(i.target())) for i in self.g.edges()])
        self.node_names = [self.g_id[i] for i in self.g.vertices()]

        # dumper(['storage objects updated'])

        # dumper(["nbr_edges new: ", str(len([i for i in self.g.edges()]))])
        # dumper(['nodes_to_add'] + nodes_to_add)
        # seems to work
        dumper(['to here'])
      
        self.recalculate_layout()
        dumper(['to here2'])


    def recalculate_layout(self):
        """calculate new change_array, set rwr_c counter"""
        dumper(['recalculating starting'])
        self.base_pos_ar = self.pos_vp.get_2d_array((0,1)).T

        # set_dict = {'p': 2, 'max_level': 20, 'adaptive_cooling': False, 
        #             'gamma': 1, 'theta': 1, 'cooling_step': 0.3, 'C': 0.6, 'mu_p': 1.2}

        # self.goal_vp = sfdp_layout(self.g, K=0.5, pos=self.pos_vp, **set_dict)
        self.goal_vp = fruchterman_reingold_layout(self.g, pos = self.pos_vp)

        
        goal_ar = self.goal_vp.get_2d_array([0,1]).T
        self.chng_ar = (goal_ar - self.base_pos_ar)/self.step
        
        self.rwr_c = self.step
        dumper(["base_pos_ar: ", self.base_pos_ar])
        dumper(["goal_ar: ", goal_ar])
        dumper(["chng_ar: ", self.chng_ar])
        dumper(['recalculating done'])


    def redraw_layout(self):
        """actually do the drawing, run multiple (step (rwr_c)) times"""
        self.cur_pos_ar = np.round(self.base_pos_ar + self.chng_ar * (self.step - self.rwr_c),3) 
        self.qt_coords = self.nolz_pos_ar(self.cur_pos_ar)

        self.rwr_c -= 1
        self.update()
        # dumper(['redrawing'])
    

    # def draw_arrow(qp, p1x, p1y, p2x, p2y):
    def draw_arrow(self, qp, p1x, p1y, p2x, p2y, node_width):
        """draw arrow from p1 to rad units before p2"""
        # get arrow angle, counterclockwise from center -> east line

        # dumper(['painting time'])
        angle = degrees(atan2((p1y-p2y), (p1x - p2x)))

        # calculate attach point
        arw_goal_x = p2x + node_width * cos(radians(angle))
        arw_goal_y = p2y + node_width * sin(radians(angle))

        # calculate start point: idk how trig works but does
        start_px = p1x - node_width * cos(radians(angle))
        start_py = p1y - node_width * sin(radians(angle))

        # arrow stuff: +/- 30 deg
        ar1 = angle + 25
        ar2 = angle - 25

        arw_len = 10

        # need to focus on vector from p2 to p1
        ar1_x = arw_goal_x + arw_len * cos(radians(ar1))
        ar1_y = arw_goal_y + arw_len * sin(radians(ar1))
                
        ar2_x = arw_goal_x + arw_len * cos(radians(ar2))
        ar2_y = arw_goal_y + arw_len * sin(radians(ar2))
        
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
        # qp.setPen(Qt.green)
        qp.setFont(QFont('Decorative', 10))
        [qp.drawText(t[0][0] + node_width, t[0][1], t[1]) for t in zip(self.qt_coords, self.node_names)]
        # dumper(['done painting'])

        qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        # qp.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        
        dumper(['painting nodes'])

        for i in zip(self.qt_coords, self.node_names):
            if self.emacs_var_dict['cur_node'] == i[1]:
                qp.setPen(QPen(Qt.black, 5, Qt.SolidLine))
                qp.drawEllipse(i[0][0]-(node_width/2), i[0][1]- (node_width/2), node_width, node_width) 
                qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))

            else:
                qp.drawEllipse(i[0][0]-(node_width/2), i[0][1]- (node_width/2), node_width, node_width)

        # qp.drawEllipse(self.c, self.c, 7, 7)
        # qp.end()

    def nolz_pos_ar(self, pos_ar_org):
        """normalize pos ar to window limits"""

        # pos_ar_org = goal_ar

        size = self.size()
        
        limits = [[20,size.width()-50], [20,size.height()-20]]

        x_max = max(pos_ar_org[:,0])
        x_min = min(pos_ar_org[:,0])
        y_max = max(pos_ar_org[:,1])
        y_min = min(pos_ar_org[:,1])

        # need linear maping function again
        pos_ar2 = pos_ar_org

        pos_ar2[:,0] = (((pos_ar2[:,0] - x_min)/(x_max - x_min)) * (limits[0][1] - limits[0][0])) + limits[0][0]
        pos_ar2[:,1] = (((pos_ar2[:,1] - y_min)/(y_max - y_min)) * (limits[1][1] - limits[1][0])) + limits[1][0]

        return(pos_ar2)



class AppBuffer(Buffer):
    def __init__(self, buffer_id, url, arguments):
        Buffer.__init__(self, buffer_id, url, arguments, True, QColor(0, 0, 0, 255))
        
        # with open("/tmp/eaf3.csv", "a") as fo:
        #     wr = csv.writer(fo)
        #     wr.writerow(["runs_now"])
        
        self.update()

        # dumper(list(self.emacs_var_dict.keys()))
        self.add_widget(ob_viz(QColor(0, 0, 0, 255)))

        self.timer = QTimer(self, timeout=self.update_demo, interval=40)
        self.timer.start()

        self.reload_timer = 0

        self.edge_string = ""
        self.cur_node = ""

        self.update()

    def update_demo(self):
        
        # initiate
        if self.buffer_widget.c == 0:
            self.buffer_widget.init2(self.emacs_var_dict)
            self.edge_string = self.emacs_var_dict['links']
            self.buffer_widget.update()
            self.cur_node = self.emacs_var_dict['cur_node']
            
        # update graph
        if self.edge_string != self.emacs_var_dict['links'] and self.buffer_widget.c > 0:
            self.buffer_widget.update_graph(self.emacs_var_dict)
            self.edge_string = self.emacs_var_dict['links']

        if self.cur_node != self.emacs_var_dict['cur_node']:
            self.buffer_widget.update()
            self.cur_node = self.emacs_var_dict['cur_node']

        # realign nodes? 
        if self.buffer_widget.rwr_c > 0:
            self.buffer_widget.redraw_layout()
            self.buffer_widget.update()

        # dumper(["update_check", self.emacs_var_dict['update_check']])
        
        if self.emacs_var_dict['update_check'] == "1" and self.reload_timer == 0:
            # self.eval_in_emacs.emit('''(message "hello")''')
            self.eval_in_emacs.emit('''(setq update_check_elisp 0)''')
            dumper(['redrawing start'])

            self.buffer_widget.init2(self.emacs_var_dict)
            self.reload_timer = 10
            self.buffer_widget.update()
            dumper(['redrawing done'])
            

        self.buffer_widget.c += 1
        if self.reload_timer > 0:
            self.reload_timer -= 1

        # dumper(["reload_timer: ", str(self.reload_timer)])

        # if self.buffer_widget.c > 1:

        #     dumper(["----------------",
        #             "c: " + str(self.buffer_widget.c), 
        #             "node count:" + str(len([i for i in self.buffer_widget.g.vertices()])),
        #             self.edge_string,
        #             self.emacs_var_dict['links'],
        #             "buffer_widget.rwr_c: " + str(self.buffer_widget.rwr_c),
        #             "----------------"
        #             ])
            
        # dumper([str(self.emacs_var_dict)])
        
        # dumper([c])
        # self.buffer_widget.drawText(10, 10, str(c))
        
                
        # link_str = self.emacs_var_dict['links']
        # dumper(self.emacs_var_dict)
        
        # links = link_str.split(";")
        # for some reason \n gets added?? maybe dumper artifact? 
        # link_tpls = [i.split(" -- ") for i in links]
        
        # link_tpls = [['possible_ares', 'organization_level_vars'],
        #              ['possible_ares', 'soccer_player_positionings'],
        #              ['possible_ares', 'music'],
        #              ['possible_ares', 'chess']]
        
        # self.adj = np.array([(int(i.source()), int(i.target())) for i in self.g.edges()])
        # self.pos_vp = sfdp_layout(self.g, K=K, max_iter=1)

            
# * scrap 
        # node_str = self.emacs_var_dict['nodes']
        # nodes = node_str.split(";")
        # dumper(nodes)
        # dumper([node_str])
