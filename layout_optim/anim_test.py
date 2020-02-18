# * example

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from random import sample, choices


from random import sample, choices

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        self.ax.axis([-10, 10, -10, 10])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        
        # hm not clear why this doesn't generate a new set of points all the time
        # xy should be called every time
        # first section only called once at beginning, then no longer
        # maybe convention? 
        print('lol')
        
        xy = (np.random.random((self.numpoints, 2))-0.5)*10
        s, c = np.random.random((self.numpoints, 2)).T
        while True:
            print('lul')
            xy += 0.03 * (np.random.random((self.numpoints, 2)) - 0.5)
            s += 0.05 * (np.random.random(self.numpoints) - 0.5)
            c += 0.02 * (np.random.random(self.numpoints) - 0.5)
            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


a = AnimatedScatter()
plt.show()


# * own test: point that moves across field

class AnimTest():
    
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.stream = self.data_stream()
        self.x = 0

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=200, 
                                          init_func=self.setup_plot, blit=True)


    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y = next(self.stream).T
        self.scat = self.ax.scatter(x, y, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        self.line, = self.ax.plot([1,2], [3,4], '-', lw=2)
        self.ax.axis([0, 10, -2, 10])
        # self.line.axes.axis([0, 10, -2, 10])

        # For FuncAnimation's sake, we need to return the artist we'll be using
        return self.scat, self.line,
    # self.scat, 

    def data_stream(self):
        # increase point 
        x = self.x
        while True:
            x = x + 0.1
            yield np.c_[[x, x],[0,2]]


    def update(self, i):
        data = next(self.stream)
        # print(data)
        self.scat.set_offsets(data)
        # print(self.ax.lines)
        # self.ax.lines = []
        self.line.set_data([data[0][0], data[1][0]], [data[0][1], data[1][1]])
        # self.line.set_xdata([data[0][0], data[1][0]])
        # self.line.set_ydata([data[0][1], data[1][1]])

        # print(self.line.get_xydata())
        # self.line.set_data([1,2], [3,4])
        # self.ax.plot(
        
        return self.scat, self.line,
    # self.scat, 
    

a = AnimTest()
# a.fig.show()
plt.show()

# seems possible to manage lines directly
# dicts would be good



# * line test

td_x = [1,2,4,6]
td_y = [1,2,3,4]

td_x = choices(np.arange(0,6, 0.1), k = 20)
td_y = choices(np.arange(0,6, 0.1), k = 20)

fig, ax = plt.subplots()
scat = ax.scatter(td_x, td_y)
line, = ax.plot([], [], 'o-', lw=2)
line.set_data([1,2], [3,4])
ax.axis([0, 6.5, 0, 6.5])

# ax.plot(test_data_x, test_data_y, 'ro-')
# [ax.plot(i[0], i[1], 'ro-')]
# first xs, then ys

[ax.plot((td_x[i[0]],td_x[i[1]]), (td_y[i[0]],td_y[i[1]])) for i in zip(choices(np.arange(len(td_x)), k=10), choices(np.arange(len(td_y)), k=10))]

fig.show()
ax.show()

plt.plot(x[i:i+2], y[i:i+2], 'ro-')

 
from random import choices


# * application: basic graph anim

class FruchtTest():
    def __init__(self, g):

        self.fig, self.ax = plt.subplots()
        self.stream = self.data_stream()
        self.g = g

        self.W = self.L = 20

        self.t = self.W/60
        self.dt = self.t/(200)
        self.area = self.W*self.L
        self.k = math.sqrt(self.W/nx.number_of_nodes(g))

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=40, 
                                           init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y = next(self.stream).T
        self.scat = self.ax.scatter(x, y, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        
        self.lines = []
        for e in g.edges():
            line, = self.ax.plot([self.g.nodes[e[0]]['x'], self.g.nodes[e[1]]['x']], 
                                 [self.g.nodes[e[0]]['y'], self.g.nodes[e[1]]['y']])
            self.lines.append(line)

        self.ax.axis([-2, 25, -2, 25])
        return self.scat, self.lines,
    
    def data_stream(self):
        
        while True: 
            
            pos = {}
            for v in self.g.nodes():
                pos[v] = [self.g.nodes[v]['x'],self.g.nodes[v]['y']]

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
                            d = f_r(delta,self.k)/delta
                            self.g.nodes[v]['dx'] += dx*d
                            self.g.nodes[v]['dy'] += dy*d

            # calculate attractive forces
            for v,u in self.g.edges():
                dx = self.g.nodes[v]['x'] - self.g.nodes[u]['x']
                dy = self.g.nodes[v]['y'] - self.g.nodes[u]['y']
                delta = math.sqrt(dx*dx+dy*dy)
                if delta != 0:
                    d = f_a(delta,self.k)/delta
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

            yield np.r_[[[self.g.nodes[v]['x'],self.g.nodes[v]['y']] for v in self.g.nodes]]

            
    
    def update(self, i):

        data = next(self.stream)

        # print(data)
        self.scat.set_offsets(data)
        # print(self.ax.lines)
        # self.ax.lines = []
        # for e in zip(g.edges, self.lines):
        for e in zip(self.g.edges, self.lines):
            e[1].set_data([g.nodes[e[0][0]]['x'], g.nodes[e[0][1]]['x']], 
                          [g.nodes[e[0][0]]['y'], g.nodes[e[0][1]]['y']])

        return self.scat, self.lines,    

f = FruchtTest(g)
plt.show()



test_el = [['a1', 'b1'],
           ['a1', 'b2'],
           ['b1', 'c1'],
           ['b2', 'c1']]

input_pos = {'a1': np.array([ 0, 2]), 
             'b1': np.array([ 1, 3]), 
             'b2': np.array([ 4 , 7]), 
             'c1': np.array([ 8, 3])}

g = nx.Graph()
g.add_edges_from(test_el)

for v in g.nodes:
    v_pos = input_pos[v]
    g.nodes[v]['x'] = v_pos[0]
    g.nodes[v]['y'] = v_pos[1]



# ** more random test

test_nodes = ['n' + str(i) for i in range(20)]
test_el2 = [i for i in [choices(test_nodes, k = 2) for i in range(60)] if i[0]!=i[1]]

g = nx.Graph()
g.add_edges_from(test_el2)

g = nx.random_geometric_graph(40, 0.2)
g = nx.random_regular_graph(3, 100)
g = nx.random_powerlaw_tree(n=30)
g = nx.random_graphs(100, 3)


for v in g.nodes:
    g.nodes[v]['x'] = choices(range(15))[0]
    g.nodes[v]['y'] = choices(range(15))[0]
    

f = FruchtTest(g)
plt.show()
# nice, overall looks as expected 
# not really efficient yet tho
# probably both plt and iterative python
# - plt: qt faster, hopefully
# - python: summarize to matrices


# should i maybe change to qt anyways already? 
# gonna be easier to implement in the end, not using any stupid plt shit in the end anyways


# ahh but i don't want to spend another day with technical implementations
# i mean qt stuff shouldn't be taking that long

# idea to use plt was kinda to be able to debug
# but that's more the iterative/custom frucht layout algorithm
# 

# * scrap
# ** multi-line test
x, y = next(self.stream).T
self.scat = self.ax.scatter(x, y, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        
lines = []
for e in g.edges():
    line, = ax.plot([g.nodes[e[0]]['x'], g.nodes[e[1]]['x']], [g.nodes[e[0]]['y'], g.nodes[e[1]]['y']])
    lines.append(line)
