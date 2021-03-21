from neo4j import GraphDatabase

drvr = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', 'anudora'))
segs = drvr.session()

import sys
from PyQt5.QtWidgets import QPushButton, QApplication, QWidget
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtDBus
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSlot

from argparse import ArgumentParser
import logging

import json

app = QApplication(sys.argv)


class obvz_window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()


        self.bus = QtDBus.QDBusConnection.sessionBus()
        self.server = QDBusServer()
        self.bus.registerObject('/cli', self.server)
        self.bus.registerService('com.qtpad.dbus')
        self.server.dbusAdaptor.message_base.connect(self.signal_received)

    def signal_received(self, string_received):
        """deal received signal"""
        logging.info('receiving')
        print(string_received)
        content = json.loads(string_received)
        print(content)

        for i in content:
            content[i] = content[i].split(" ")

        print(content)
        
        # main parsing/updates has to happen here

# update check with test children (tchin)
tchin1 = ['wrk1', 'wrk2', 'wrk3']
tchin2 = ['wrk1', 'wrk2', 'wrk3', 'wrk4', 'wrk5']
tchin3 = ['wrk1', 'wrk3', 'wrk5']

params1 = {'pairs': [{'tag': 'test', 'work': i} for i in tchin1]}
params2 = {'pairs': [{'tag': 'test', 'work': i} for i in tchin2]}
params3 = {'pairs': [{'tag': 'test', 'work': i} for i in tchin3]}

node_params = {'nodes': [{'name': 'test'}]}

del_query = """UNWIND $nodes as node
match (f:tag {name: node.name})-[r:CONCERNS]->(n) delete r
"""


tag_qry = """UNWIND $pairs as pair
MERGE (a:tag {name: pair.tag})
WITH a, pair
MERGE (b:work {name: pair.work})
MERGE (a)-[:CONCERNS]->(b)"""

x=segs.run(del_query, parameters = node_params)
x= segs.run(tag_qry, parameters = params1)

x=segs.run(del_query, parameters = node_params)
x= segs.run(tag_qry, parameters = params2)

x=segs.run(del_query, parameters = node_params)
x= segs.run(tag_qry, parameters = params3)





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
