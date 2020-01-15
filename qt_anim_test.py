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
        self.width = 500
        self.height = 500
        self.InitWindow()
        self.g = g
        self.W = self.L = 20
        
        self.t = self.W/60
        self.dt = self.t/(200)
        self.area = self.W*self.L
        self.k = math.sqrt(self.W/nx.number_of_nodes(g))

        self.timer = QTimer(self, timeout=self.update_demo, interval=40)
        self.timer.start()
        
        
    def f_a(d,k):
        return d*d/k

    # repulsive force
    def f_r(d,k):
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


        # pos = {}
        # for v in self.g.nodes():
        #     pos[v] = [self.g.nodes[v]['x'],self.g.nodes[v]['y']]

        # for v in self.g.nodes:
        #     self.g.nodes[v]['x'] += 0.1
        #     self.g.nodes[v]['y'] += 0.1

        # yield np.r_[[[self.g.nodes[v]['x'],self.g.nodes[v]['y']] for v in self.g.nodes]]

    

    
