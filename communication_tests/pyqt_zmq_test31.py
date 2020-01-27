
import sys
import zmq
from time import sleep, time

from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer


from PyQt5 import QtCore, QtGui, QtWidgets

                
class ZeroMQ_Listener(QtCore.QObject):

    message = QtCore.pyqtSignal(str)
    
    def __init__(self):
        print('init1')
        QtCore.QObject.__init__(self)
        
        # Socket to talk to server
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)

        print("Collecting updates from weather server")
        self.socket.connect ("tcp://localhost:5556")
        print('init2')
        # Subscribe to zipcode, default is NYC, 10001
        filter = str(app.arguments()[1]) if len(app.arguments()) > 1 else "10001"
        # self.socket.setsockopt(zmq.SUBSCRIBE, filter)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        self.running = True
        self.ctr = 0
        print('init3')

    def loop(self):
        while self.running:
            string = self.socket.recv()
            print(string)
            # for i in range(10):
            #     self.message.emit(str(string, 'utf-8'))
            #     sleep(1)
            self.message.emit(str(string, 'utf-8'))

            
            # do the update stuff here? 
            # that would be so weird


class ZeroMQ_Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # QtWidgets.QMainWindow.__init__(self, parent)
        # frame = QtWidgets.QFrame()
        # label = QtWidgets.QLabel("listening")
        # self.text_edit = QtWidgets.QTextEdit()
        
        # layout = QtWidgets.QVBoxLayout(frame)
        # layout.addWidget(label)
        # layout.addWidget(self.text_edit)
        
        # self.setCentralWidget(frame)
        
        self.title = "PyQt5 Drawing Tutorial"
        self.top= 150
        self.left= 150
        self.width = 500
        self.height = 500
        self.InitWindow()
        self.x = 5

        self.thread = QtCore.QThread()
        self.zeromq_listener = ZeroMQ_Listener()
        self.zeromq_listener.moveToThread(self.thread)
        self.thread.started.connect(self.zeromq_listener.loop)
        self.ctr = 0
        self.paint_timer = QtCore.QTimer(self, timeout=self.timer_func, interval=100)
        
        self.zeromq_listener.message.connect(self.signal_received)

        QtCore.QTimer.singleShot(0, self.thread.start)
        # self.show()
        # for i in range(10):
        #     # print('kk') 
        #     self.text_edit.append("%s\n"% "kk")
        #     self.update()
        #     sleep(0.3)


    # def stop_timer(self):
    #     print('stop paint timer')
    #     self.viszlr.paint_timer.stop()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.green,  8, Qt.DashLine))
        painter.drawEllipse(self.x, 40, 10, 10)


    
    def timer_func(self):
        print('lol')
        # self.text_edit.append("%s\n"% str(self.ctr))
        self.ctr += 1
        print(self.ctr)
        self.x +=3
        self.update()

        if self.ctr == 10:
            self.paint_timer.stop()
            self.ctr = 0

    
    def signal_received(self, message):
        print('signal receiving')
        # sleep(2)
        print('first message appending')
        # self.text_edit.append("%s\n"% message)
        self.paint_timer.start()

        print(message)

    def closeEvent(self, event):
        self.zeromq_listener.running = False
        self.thread.quit()
        self.thread.wait()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    mw = ZeroMQ_Window()
    # mw.show()

    sys.exit(app.exec_())
    

