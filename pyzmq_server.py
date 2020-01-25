#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")
socket.bind("tcp://127.0.0.1:5555")

while True:
    #  Wait for next request from client
    print('lol')
    
    message = socket.recv()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(0.2)

    #  Send reply back to client
    socket.send(b"World")
