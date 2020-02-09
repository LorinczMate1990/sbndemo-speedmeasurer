#!/usr/bin/python
# -*- coding: utf-8 -*-

import zmq


port = 5556

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

while True:
    try:
        message = socket.recv_pyobj(zmq.NOBLOCK)
        messageGot = True
    except zmq.Again:
        messageGot = False
        
    if messageGot:
            print message
