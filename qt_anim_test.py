from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont

import sys
import signal

import math

import numpy as np
from random import sample, choices

from collections import Counter

import networkx as nx


class QtTest(QWidget):
    def __init__(self, g):
        super().__init__()
        self.title = "PyQt5 Drawing Tutorial"
        self.top= 150
        self.left= 150
        self.width = 800
        self.height = 800
        self.InitWindow()
        self.g = g
        self.W = self.L = 1200
        self.area = self.W*self.L

        self.t = self.W/60
        self.dt = self.t/(200)
        self.area = self.W*self.L
        # k is something with strength of repulsive force
        self.k = math.sqrt((self.area)/nx.number_of_nodes(g))/4
        print('k: ', self.k)
        self.timer = QTimer(self, timeout=self.update_demo, interval=40)
        self.timer.start()
        
        
    def f_a(self, d,k):
        return d*d/k

    # repulsive force
    def f_r(self, d,k):
        return k*k/d

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def update_demo(self):
        """update positions, save in self.g vertex properties directly (no return)"""
        # pos = {}
        # for v in self.g.nodes():
        #     pos[v] = [self.g.nodes[v]['x'],self.g.nodes[v]['y']]

        for v in self.g.nodes():
            self.g.nodes[v]['dx'] = 0
            self.g.nodes[v]['dy'] = 0
            for u in self.g.nodes():
                if v != u:
                    dx = self.g.nodes[v]['x'] - self.g.nodes[u]['x']
                    dy = self.g.nodes[v]['y'] - self.g.nodes[u]['y']
                    # use own distance function here
                    delta = math.sqrt(dx*dx+dy*dy)
                    if delta != 0:
                        d = self.f_r(delta,self.k)/delta
                        self.g.nodes[v]['dx'] += dx*d
                        self.g.nodes[v]['dy'] += dy*d

        # calculate attractive forces
        for v,u in self.g.edges():
            dx = self.g.nodes[v]['x'] - self.g.nodes[u]['x']
            dy = self.g.nodes[v]['y'] - self.g.nodes[u]['y']
            delta = math.sqrt(dx*dx+dy*dy)
            if delta != 0:
                d = self.f_a(delta,self.k)/delta
                ddx = dx*d
                ddy = dy*d
                self.g.nodes[v]['dx'] += -ddx
                self.g.nodes[u]['dx'] += +ddx
                self.g.nodes[v]['dy'] += -ddy
                self.g.nodes[u]['dy'] += +ddy

        # limit the maximum displacement to the temperature t
        # and then prevent from being displace outside frame
        for v in self.g.nodes():
            dx = self.g.nodes[v]['dx']
            dy = self.g.nodes[v]['dy']
            disp = math.sqrt(dx*dx+dy*dy)
            if disp != 0:
                # cnt += 1
                d = min(disp,self.t)/disp
                x = self.g.nodes[v]['x'] + dx*d
                y = self.g.nodes[v]['y'] + dy*d
                x =  min(self.W,max(0,x)) - self.W/2
                y =  min(self.L,max(0,y)) - self.L/2
                self.g.nodes[v]['x'] = min(math.sqrt(self.W*self.W/4-x*x),max(-math.sqrt(self.W*self.W/4-x*x),x)) + self.W/2
                self.g.nodes[v]['y'] = min(math.sqrt(self.L*self.L/4-y*y),max(-math.sqrt(self.L*self.L/4-y*y),y)) + self.L/2
                # not clear what's happening here
                # but it sure looks inefficient
                # maybe make repulsive force at border? 


        # cooling
        self.t -= self.dt
        print('temperature: ', self.t)
        if self.t < 0: 
            self.t = 0

        self.update()

        # pos = {}
        # for v in self.g.nodes():
        #     pos[v] = [self.g.nodes[v]['x'],self.g.nodes[v]['y']]

        # for v in self.g.nodes:
        #     self.g.nodes[v]['x'] += 0.1
        #     self.g.nodes[v]['y'] += 0.1

        # yield np.r_[[[self.g.nodes[v]['x'],self.g.nodes[v]['y']] for v in self.g.nodes]]

    def paintEvent(self, event):

        node_width = 10
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        # edges = [(self.qt_coords[i[0]], self.qt_coords[i[1]]) for i in self.adj]
        edges = [[(self.g.nodes[e[0]]['x'], self.g.nodes[e[0]]['y']),
                  (self.g.nodes[e[1]]['x'], self.g.nodes[e[1]]['y'])]
            for e in self.g.edges]
        
        # dumper([str(i) for i in edges])

        qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))

        # [qp.drawLine(e[0][0], e[0][1], e[1][0], e[1][1]) for e in edges]
        [self.draw_arrow(qp, e[0][0], e[0][1], e[1][0], e[1][1], (node_width/2) + 5) for e in edges]
        
        qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        # [qp.drawEllipse(g.nodes[v]['x']-(node_width/2), g.nodes[v]['y']- (node_width/2), node_width, node_width) for v in g.nodes]
        [qp.drawRect(g.nodes[v]['x'] - (g.nodes[v]['width']/2), 
                     g.nodes[v]['y'] - (g.nodes[v]['height']/2), 
                     g.nodes[v]['width'], 
                     g.nodes[v]['height']) for v in g.nodes]


    
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
        
        print(start_px, start_py, arw_goal_x, arw_goal_y)

        # qp.drawLine(p1x, p1y, p2x, p2y)
        # qp.drawLine(p1x, p1y, arw_goal_x, arw_goal_y)
        qp.drawLine(start_px, start_py, arw_goal_x, arw_goal_y)
        qp.drawLine(ar1_x, ar1_y, arw_goal_x, arw_goal_y)
        qp.drawLine(ar2_x, ar2_y, arw_goal_x, arw_goal_y)

if __name__ == "__main__":
    g = nx.random_geometric_graph(20, 0.3)

    for v in g.nodes:
        g.nodes[v]['x'] = choices(range(15))[0]
        g.nodes[v]['y'] = choices(range(15))[0]

        g.nodes[v]['width'] = g.nodes[v]['height'] = choices(range(15))[0]
    
    App = QApplication(sys.argv)
    window = QtTest(g)
    App.exec()
