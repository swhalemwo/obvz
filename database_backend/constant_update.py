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


rel_type_dict = {"BRAIN_CHILDREN": 'bc',
                 "BRAIN_PARENTS" : 'bp'}


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
        # print(string_received)
        content = json.loads(string_received)
        # print(content)

        # edge separate edge_list for each link type

        edge_list = self.process_received_content(content)

        for rel_type in rel_type_dict.values():
            if rel_type in edge_list:
                self.update_n4j_entries(edge_list[rel_type], rel_type)
        

    def process_received_content(self, content):
        edge_list = {}

        # k = list(content.keys())[0] # this is ugly way to get name, and won't work on 

        # for coerg in ["BRAIN_CHILDREN", "BRAIN_PARENTS"]:

        for k,v in content.items():
            if v is not None:
                for coerg,nodes_string in v.items():
                    if coerg in ["BRAIN_CHILDREN", "BRAIN_PARENTS"]:
                        coerg_short = rel_type_dict[coerg]

                        nodes_split = nodes_string.split(" ")
                        # print(coerg, "\n")
                        # print(nodes_split)

                        if coerg_short in edge_list:
                            for j in nodes_split:
                                edge_list[coerg_short].append([k, j])
                        else:
                            edge_list[coerg_short] = [[k,j] for j in nodes_split]

        print(edge_list)
        return edge_list 


        # contents = [{k: v.split(" ") for k,v in i.items()} for i in content.items()]

        # print(content)

        # for k,v in content.items():
            
        
        # main parsing/updates has to happen here

    def update_n4j_entries(self, edge_list, rel_type):
        """would be good to have general approach that can handle any number of entries"""
        # should be two node types: yeah but created here

        # edge list: just list of lists?


        node1_params = {'nodes': [{'name': i} for i in list(set([k[0] for k in edge_list]))]}

        # first del query to clear up
        del_query = """UNWIND $nodes as node
        match (f {name: node.name})-[r:""" + rel_type + """]->(n) delete r
        """
        print(del_query)

        edge_params = {'pairs': [{'node1': i[0], 'node2': i[1]} for i in edge_list]}

        add_qry = """UNWIND $pairs as pair
        MERGE (a {name: pair.node1})
        WITH a, pair
        MERGE (b {name: pair.node2})
        MERGE (a)-[:""" + rel_type + """]->(b)"""


        x=segs.run(del_query, parameters = node1_params)
        x= segs.run(add_qry, parameters = edge_params)


    
# keywords = {'BRAIN_CHILDREN': ['Molnar_2005_architecture', 'Komarova_Velthuis_2018_activation', 'Warczok_Beyer_2021_sociology'], 'BRAIN_PARENTS': ['cls_tags']}

# # keywords edge_list 
# kw_el = [['field', i] for i in keywords['BRAIN_CHILDREN']]

# update_n4j_entries(kw_el, 'bc')



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

    
