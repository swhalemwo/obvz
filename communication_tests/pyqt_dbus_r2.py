#!/usr/bin/python3
from PyQt5 import QtCore, QtDBus
from PyQt5.QtCore import QObject, pyqtSlot

from PyQt5.QtWidgets import QApplication, QWidget



import sys
 
class QDBusServer(QObject):
    
    message = QtCore.pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self)
        self.dbusAdaptor = QDBusServerAdapter(self)
        # self.message.connect(self.func2)
        # self.dbusAdaptor.message.connect(self.signal_received)
 
    # def signal_received(self, some_str):
    #     print('gotcha' + some_str)

    def func2(self):
        print('parent time')
        self.message.emit('ddd')
        # self.parent().sig2()
 
class QDBusServerAdapter(QtDBus.QDBusAbstractAdaptor):
    
    QtCore.Q_CLASSINFO("D-Bus Interface", "com.qtpad.dbus")
    QtCore.Q_CLASSINFO("D-Bus Introspection",
    '  <interface name="com.qtpad.dbus">\n'
    '    <property name="name" type="s" access="read"/>\n'
    '    <method name="echo">\n'
    '      <arg direction="in" type="s" name="phrase"/>\n'
    '    </method>\n'
    '  </interface>\n')
 
    message = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        # self.setAutoRelaySignals(True)


    @pyqtSlot(str, result=str)
    def echo(self, phrase):
        self.message.emit(str(phrase))
        print("parse(" + phrase + ")")
        self.parent().func2()

class main_prog(QWidget):
    
    bus = QtDBus.QDBusConnection.sessionBus()
    server = QDBusServer()
    bus.registerObject('/cli', server)
    bus.registerService('com.qtpad.dbus')

    def __init__(self):
        super().__init__()
        # QWidget.__init__(self)
        self.counter = 0

    
        # self.server.message.connect(self.sig2)
        self.server.message.connect(self.signal_received)


        self.paint_timer = QtCore.QTimer(self, timeout=self.timer_func, interval=200)
        self.paint_timer.start()
        print('inited')


        
    def timer_func(self):
        print(self.counter)
        self.counter +=1
    

    # def sig2(self):
    #     print('back in main')
    #     self.counter = 0

    def signal_received(self, some_str):
        print('gotcha')
        self.counter = 0
        # self.increase_ctr()

    # def increase_ctr(self):
    #     print('time to increase counter')
    #     print(str(self.counter))
    #     self.counter +=1



if __name__ == '__main__':
    # app = QtCore.QCoreApplication(sys.argv)
    app = QApplication(sys.argv)

    x = main_prog()

    print(type(x))

    

    sys.exit(app.exec_())
