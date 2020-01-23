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

App = QApplication(sys.argv)

def mat_scaler(A):
    """scales up square matrix by doubling each dimension + reordering, effectively splitting each cell into 4"""
    A_horz_list = [A]*4
    A_horz_arr = np.hstack(A_horz_list)
    A_vert_list = [A_horz_arr]*4
    A_scl = np.vstack(A_vert_list)
    
    nbr_points = A.shape[0]
    A_order = [(i, i + nbr_points, i + 2*nbr_points, i + 3*nbr_points) for i in range(nbr_points)]
    A_order_flat = [item for sublist in A_order for item in sublist]
        
    A_scl_rorder = A_scl[np.ix_(A_order_flat, A_order_flat)]

    return(A_scl_rorder)

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





class QtTest(QWidget):
    def __init__(self, g):
        super().__init__()
        self.title = "PyQt5 Drawing Tutorial"
        self.top= 0
        self.left= 0
        self.width = 800
        self.height = 800
        self.InitWindow()
        self.g = g
        self.W = self.L = 800
        self.area = self.W*self.L

        self.t = self.W/60
        self.dt = self.t/(200)
        self.area = self.W*self.L

        self.main_v = ""
        self.cmpr_v = ""
        # k is something with strength of repulsive force
        
        # self.k = math.sqrt((self.area)/nx.number_of_nodes(g))/4
        self.k = 50
        print('k: ', self.k)
        self.timer = QTimer(self, timeout=self.update_demo, interval=40)
        self.timer.start()
        
        
    def f_a(self, d,k):
        return d*d/k

    # repulsive force
    def f_r(self, d,k):
        return k*k/d

    def sq_dist(self, sq1, sq2):
        """distance between two squares"""
        # sq1 = np.array([1,1])
        # sq2 = np.array([5,4])

        sq1_tpl = np.array([self.g.nodes[sq1]['x'], self.g.nodes[sq1]['y']])
        sq2_tpl = np.array([self.g.nodes[sq2]['x'], self.g.nodes[sq2]['y']])

        # just one vector/angle needed due to symmetry: hyps same if 180 deg turned
        vec= sq2_tpl-sq1_tpl
        angle = math.degrees(math.atan2(vec[0], vec[1]))

        hyp1 = self.sq_hyp(sq1_tpl, angle, self.g.nodes[sq1]['height']/2)
        hyp2 = self.sq_hyp(sq2_tpl, angle, self.g.nodes[sq1]['height']/2)

        dist_ctrs = np.sqrt(vec[0]**2 + vec[1]**2)
        dist_ttl = dist_ctrs- hyp1 - hyp2
        dist_ttl_crct = np.clip(dist_ttl, 0, None)
        return(dist_ttl_crct)


    def sq_hyp(self, sq, angle, adj):
        """gets the hypotenuse of the triangle (distance to border)"""
        # needs to be generalized to rectangle
        angle = angle % 90

        # adj = 1

        hyp = adj/math.cos(math.radians(angle))
        return(hyp)
    
    # idk maybe easier to use actual shortest distances (not through centers)
    # more straightforward
    # covers all rectangles


    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def dim_ovlp(self, rng1, rng2):
        
        # rng1_set = set(np.range(rng1[0], rng1[1], 0.1), 2))
        # rng2_set = set(np.range(rng2[0], rng2[1], 0.1), 2))
        
        rng1_set = set(range(int(np.floor(rng1[0])), int(np.ceil(rng1[1]))))
        rng2_set = set(range(int(np.floor(rng2[0])), int(np.ceil(rng2[1]))))

        intsection = rng1_set.intersection(rng2_set)
        return(intsection)

    def rect_points(self,r):
        p1 = [r[0] + (r[2]/2), r[1] + (r[3]/2)]
        p2 = [r[0] + (r[2]/2), r[1] - (r[3]/2)]
        p3 = [r[0] - (r[2]/2), r[1] + (r[3]/2)]
        p4 = [r[0] - (r[2]/2), r[1] - (r[3]/2)]

        return np.array([p1, p2, p3, p4])

        
    def sq_dist2(self, sq1, sq2):
        """get distances between rects"""
        rct1 = [self.g.nodes[sq1]['x'], self.g.nodes[sq1]['y'], self.g.nodes[sq1]['width'], self.g.nodes[sq1]['height']]
        rct2 = [self.g.nodes[sq2]['x'], self.g.nodes[sq2]['y'], self.g.nodes[sq2]['width'], self.g.nodes[sq2]['height']]

        # rct2 = [1,5,2,3]
        # rct1 = [4,5,1,1]
        # rct1 =  [704.85, 334.94, 155, 197]
        # rct2 =  [605, 229, 68, 167]



        # get ranges of rectangles
        rng1x = ((rct1[0]-rct1[2]/2), (rct1[0]+rct1[2]/2))
        rng2x = ((rct2[0]-rct2[2]/2), (rct2[0]+rct2[2]/2))

        rng1y = ((rct1[1]-rct1[3]/2), (rct1[1]+rct1[3]/2))
        rng2y = ((rct2[1]-rct2[3]/2), (rct2[1]+rct2[3]/2))

        ovlp_x = self.dim_ovlp(rng1x, rng2x)
        ovlp_y = self.dim_ovlp(rng1y, rng2y)

        if ovlp_x == set() and ovlp_y == set():
            # print('no overlap whatsoever')

            rct_pts1 = self.rect_points(rct1)
            rct_pts2 = self.rect_points(rct2)
            dist_ar = euclidean_distances(rct_pts1, rct_pts2)
            min_dist = np.min(dist_ar)
            min_loc = np.where(dist_ar == min_dist)

            pt1 = rct_pts1[min_loc[0]]
            pt2 = rct_pts2[min_loc[1]]

            dx = pt1[0][0] - pt2[0][0]
            dy = pt1[0][1] - pt2[0][1]

        elif ovlp_x == set() and len(ovlp_y) > 0:
            # print('overlap in y, not x')
            rct1_pts = np.array(rng1x)
            rct2_pts = np.array(rng2x)

            dist_ar = np.array([
                [abs(rct1_pts[0] - rct2_pts[0]), abs(rct1_pts[0] - rct2_pts[1])],
                [abs(rct1_pts[1] - rct2_pts[0]), abs(rct1_pts[1] - rct2_pts[1])]])
            min_dist = np.min(dist_ar)
            min_loc = np.where(dist_ar == min_dist)

            pt1 = rct1_pts[min_loc[0][0]]
            pt2 = rct2_pts[min_loc[1][0]]

            dx = pt1 - pt2
            dy = 0

        elif ovlp_y == set() and len(ovlp_x) > 0:
            # print('overlap in x, not y')
            # overlap in y dimension but not x
            rct1_pts = np.array(rng1y)
            rct2_pts = np.array(rng2y)

            dist_ar = np.array([
                [abs(rct1_pts[0] - rct2_pts[0]), abs(rct1_pts[0] - rct2_pts[1])],
                [abs(rct1_pts[1] - rct2_pts[0]), abs(rct1_pts[1] - rct2_pts[1])]])

            min_dist = np.min(dist_ar)
            min_loc = np.where(dist_ar == min_dist)

            # print(rct2_pts)
            # print(rct1_pts)

            # print(dist_ar)
            # print(min_dist)
            # print(min_loc)
                  
            pt1 = rct1_pts[min_loc[0][0]]
            pt2 = rct2_pts[min_loc[1][0]]

            dy = pt1 - pt2
            dx = 0

        else:
            print('overlap in both')
            dx = (rct1[0] - rct2[0]) * 0.2
            dy = (rct1[1] - rct2[1]) * 0.2


        return dx, dy

    def rect_points(self, r):
        p1 = [r[0] + (r[2]/2), r[1] + (r[3]/2)]
        p2 = [r[0] + (r[2]/2), r[1] - (r[3]/2)]
        p3 = [r[0] - (r[2]/2), r[1] + (r[3]/2)]
        p4 = [r[0] - (r[2]/2), r[1] - (r[3]/2)]

        return np.array([p1, p2, p3, p4])


    def update_demo(self):
        """update positions, save in self.g vertex properties directly (no return)"""
        # pos = {}
        # for v in self.g.nodes():
        #     pos[v] = [self.g.nodes[v]['x'],self.g.nodes[v]['y']]

        
        sqs = []
        for i in self.g.nodes():
            sqx = self.rect_points([self.g.nodes[i]['x'], self.g.nodes[i]['y'], 
                                    self.g.nodes[i]['width'], self.g.nodes[i]['height']])
            sqs.append(sqx)

        # sqs = []
        # for i in w.g.nodes():
        #     sqx = w.rect_points([w.g.nodes[i]['x'], w.g.nodes[i]['y'], 
        #                             w.g.nodes[i]['width'], w.g.nodes[i]['height']])
        #     sqs.append(sqx)


        pos = np.concatenate(sqs)
        A = nx.to_numpy_array(self.g)
        # A = nx.to_numpy_array(self.g)
        
        pos_nds = np.array([[self.g.nodes[i]['x'], self.g.nodes[i]['y']] for i in self.g.nodes])
        # pos_nds = np.array([[w.g.nodes[i]['x'], w.g.nodes[i]['y']] for i in w.g.nodes])
        
        delta_nds = pos_nds[:, np.newaxis, :] - pos_nds[np.newaxis, :, :]

        nbr_nds = A.shape[0]
        nbr_pts = pos.shape[0]

        row_order = get_point_mat_reorder_order(len(self.g.nodes))
        # row_order = get_point_mat_reorder_order(len(w.g.nodes))
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

        # all_ovlp_test = x_ovlp2 + y_ovlp2 + both_ovlp + none_ovlp

        distance = (x_ovlp2 * dy_min) + (y_ovlp2 * dx_min) + (both_ovlp * 1) + (none_ovlp * min_pt_dists)
        np.clip(distance, 1, None, out = distance)
        
        displacement = np.einsum('ijk,ij->ik',
                                 delta_nds,
                                 (self.k * self.k / distance**2) - A * distance / self.k)

        # displacement = np.einsum('ijk,ij->ik',
        #                          delta_nds,
        #                          (k * k / distance**2) - A * distance / k)

        dim_ar = np.array([[g.nodes[i]['width'], g.nodes[i]['height']] for i in g.nodes])

        # repellant edges
        displacement[:,0] += self.k**2/(pos_nds[:,0] - dim_ar[:,0]/2)**2
        displacement[:,1] += self.k**2/(pos_nds[:,1] - dim_ar[:,1]/2)**2
        displacement[:,0] -= self.k**2/(self.width - (pos_nds[:,0] + dim_ar[:,0]/2))**2
        displacement[:,1] -= self.k**2/(self.height - (pos_nds[:,1] + dim_ar[:,1]/2))**2


        
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum('ij,i->ij', displacement, self.t / length)

        
        # re-assigning values
        for i in zip(self.g.nodes, delta_pos):
            self.g.nodes[i[0]]['x'] += i[1][0]
            self.g.nodes[i[0]]['y'] += i[1][1]


        # maybe add repellant edges again here
        # maybe optimize it too

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
        [self.draw_arrow(qp, e[0][0], e[0][1], e[1][0], e[1][1], (node_width/2) + 5) for e in edges]
        # [qp.drawLine(e[0][0], e[0][1], e[1][0], e[1][1]) for e in edges]
        
        # qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))

        # debug overlap
        
        qp.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        for v in g.nodes:
            
            # if v == self.main_v:
            #     qp.setPen(QPen(Qt.blue, 3, Qt.SolidLine))
                
            # if v == self.cmpr_v:
            #     qp.setPen(QPen(Qt.blue, 3, Qt.SolidLine))
                
            # else:
            qp.drawRect(g.nodes[v]['x'] - (g.nodes[v]['width']/2), 
                     g.nodes[v]['y'] - (g.nodes[v]['height']/2), 
                     g.nodes[v]['width'], 
                     g.nodes[v]['height'])

        qp.setFont(QFont('Decorative', 10))
        [qp.drawText(g.nodes[v]['x'], g.nodes[v]['y'], str(v)) for v in self.g.nodes]



        # [qp.drawRect(g.nodes[v]['x'] - (g.nodes[v]['width']/2), 
        #      g.nodes[v]['y'] - (g.nodes[v]['height']/2), 
        #      g.nodes[v]['width'], 
        #      g.nodes[v]['height']) for v in g.nodes]


    
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
        
        # print(start_px, start_py, arw_goal_x, arw_goal_y)

        # qp.drawLine(p1x, p1y, p2x, p2y)
        # qp.drawLine(p1x, p1y, arw_goal_x, arw_goal_y)
        qp.drawLine(start_px, start_py, arw_goal_x, arw_goal_y)
        qp.drawLine(ar1_x, ar1_y, arw_goal_x, arw_goal_y)
        qp.drawLine(ar2_x, ar2_y, arw_goal_x, arw_goal_y)

if __name__ == "__main__":
    

    while True:
        g = nx.random_geometric_graph(20, 0.25)
        # g = nx.random_geometric_graph(2, 1)
        # if nx.number_connected_components(g) == 2 and min([len(i) for i in nx.connected_components(g)]) > 5:
        if nx.number_connected_components(g) == 1: 
            break

    for v in g.nodes:
        g.nodes[v]['x'] = choices(range(100, 700))[0]
        g.nodes[v]['y'] = choices(range(100, 700))[0]

        g.nodes[v]['width'] = choices(range(10,75))[0]
        g.nodes[v]['height'] = choices(range(10,75))[0]

    # g.nodes[0]['x'] = g.nodes[0]['y'] = 100
    # g.nodes[1]['x'] = g.nodes[1]['y'] = 110
    
    # g.nodes[0]['width'] = g.nodes[0]['height'] = 10
    # g.nodes[1]['width'] = g.nodes[1]['height'] = 10
    
    
    

    w = QtTest(g)
    
    # w.timer.stop()
    
    App.exec()

    
# think i have to add angle 
# 

