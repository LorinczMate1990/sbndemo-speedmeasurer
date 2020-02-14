#!/usr/bin/python
# -*- coding: utf-8 -*-

import zmq
import time
import signal

port = 5556
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % port)

class InterrputException(BaseException): pass

def interrupted(signum, frame):
    "called when read times out"
    raise InterrputException()
    
signal.signal(signal.SIGALRM, interrupted)

actSpeed = 0

def timedInput():
    global actSpeed
    signal.alarm(1) # Timeout másodpercben
    try:
        foo = input()
        actSpeed = int(foo)
    except InterrputException:
        #timeout
        pass

while True:
    socket.send_pyobj({'distance':1,'dTime':1, 'speed':actSpeed})
    timedInput()
