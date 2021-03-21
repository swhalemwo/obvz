# from neo4j import GraphDatabase

# drvr = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', 'anudora'))
# segs = drvr.session()

import sys
from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtDBus
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSlot

from argparse import ArgumentParser
import logging

app = QApplication(sys.argv)


class obvz_window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()


        self.bus = QtDBus.QDBusConnection.sessionBus()
        self.server = QDBusServer()
        self.bus.registerObject('/cli', self.server)
        self.bus.registerService('com.qtpad.dbus')
        self.server.dbusAdaptor.message_base.connect(self.signal_received)

    def signal_received(self, new_graph_str):
        """deal received signal"""
        logging.info('receiving')
        # main parsing/updates has to happen here

# ---------- dbus connection start -------

class QDBusServer(QObject):

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
        # print("phrase: " + phrase + " received in Adaptor object")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-v', default=False, action='store_true')

    args = parser.parse_args()


    if args.v == True:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    mw = obvz_window()
    sys.exit(app.exec_())
