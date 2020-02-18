#!/usr/bin/python3
from PyQt5 import QtCore, QtDBus
from PyQt5.QtCore import QObject, pyqtSlot

from PyQt5.QtWidgets import QApplication, QWidget



import sys
 
class QDBusServer(QObject):
    
    # message = QtCore.pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self)
        self.dbusAdaptor = QDBusServerAdapter(self)

        
class QDBusServerAdapter(QtDBus.QDBusAbstractAdaptor):
    message_base = QtCore.pyqtSignal(str)
   
    QtCore.Q_CLASSINFO("D-Bus Interface", "com.qtpad.dbus")
    QtCore.Q_CLASSINFO("D-Bus Introspection",
    '  <interface name="com.qtpad.dbus">\n'
    '    <property name="name" type="s" access="read"/>\n'
    '    <method name="echo">\n'
    '      <arg direction="in" type="s" name="phrase"/>\n'
    '    </method>\n'
    '  </interface>\n')
 

    def __init__(self, parent):
        super().__init__(parent)
        
    @pyqtSlot(str, result=str)
    def echo(self, phrase):
        self.message_base.emit(phrase)
        print("phrase: " + phrase + " received in Adaptor object")
        

class main_prog(QWidget):
    

    def __init__(self):
        super().__init__()
        self.counter = 0

        self.bus = QtDBus.QDBusConnection.sessionBus()
        self.server = QDBusServer()
        self.bus.registerObject('/cli', self.server)
        self.bus.registerService('com.qtpad.dbus')

    
        self.server.dbusAdaptor.message_base.connect(self.signal_received)

        self.paint_timer = QtCore.QTimer(self, timeout=self.timer_func, interval=200)
        self.paint_timer.start()
        print('inited')

        
    def timer_func(self):
        print(self.counter)
        self.counter +=1
    

    def signal_received(self, some_str):
        print('phrase ' + some_str + ' received in main object')
        self.counter = 0



if __name__ == '__main__':
    # app = QtCore.QCoreApplication(sys.argv)
    app = QApplication(sys.argv)

    x = main_prog()

    sys.exit(app.exec_())
