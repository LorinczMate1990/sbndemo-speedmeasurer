#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2 
import matplotlib.pyplot as plt
import cv2.aruco as aruco
import sys
import time
import zmq
from scipy import optimize

np.set_printoptions(2**31) # If I print an array, I would like to see it...

def convertFromScreenToFloorCoordSystem(x, y, M):
    """ 
    x and y are in screen coordinates,
    M is the return value of getPerspectiveTransform
    """
    M_inv = np.linalg.inv(M)
    
    X = (M_inv[0,0]*x + M_inv[0,1]*y + M_inv[0,2]) / (M_inv[2,0]*x + M_inv[2,1]*y + M_inv[2,2])
    Y = (M_inv[1,0]*x + M_inv[1,1]*y + M_inv[1,2]) / (M_inv[2,0]*x + M_inv[2,1]*y + M_inv[2,2])
    
    return X, Y

def getSkeleton(img):
    " The input is a binary image. "
    originalInput = img # I don't want to modify the original image
    img = img.copy()

    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    # I added an attitional step, because the original algorithm keeps parts of the contours
    # I make a dilation, so the got contours will be out of the original shape
    # After the last step, I can eliminate the contours by a simple logical or
    img = cv2.dilate(img, np.ones((3,3)), iterations = 5 )
    
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            break
    
    return cv2.bitwise_and(skel, originalInput)

def getSkeleton(img):
    " The input is a binary image. "
    originalInput = img # I don't want to modify the original image
    prevImg = img.copy()
    img = img.copy()

    while np.nonzero(img)[0].size > 0:
        prevImg = img.copy()
        img = cv2.erode(img, np.ones((3,3)))
    
    return prevImg

def getMassCenter(img):
    nonzeros = np.nonzero(img)
    x = nonzeros[1]
    y = nonzeros[0]
    if np.size(x)>0:
        return int(x.mean()), int(y.mean())
    else:
        return None, None       
    
def getNonzeroPixels(img):
    nonzeros = np.nonzero(img)
    X = nonzeros[1]
    Y = nonzeros[0]
    for x, y in zip(X, Y):
        yield x, y 
    
def getDistanceSquare(a, b):
    return math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2)

show = 0
pause = False
calculateCircle = False
refreshAruco = True

if __name__ == '__main__':
    useCamera = "simulatorport" not in sys.argv
    manualSpeed = "manualspeed" in sys.argv
    manualSpeedValue = 0
    corners = []
    if not useCamera:
        i = sys.argv.index('simulatorport')
        simulatorPort = sys.argv[i+1]
        
    useDefaultArucoSize = "aruco" not in sys.argv
    if not useDefaultArucoSize:
        i = sys.argv.index('aruco')
        aruco_dim = sys.argv[i+1]
    else:
        aruco_dim = 860-170

    context = zmq.Context()
        
    if not useCamera:
        simulatorSocket = context.socket(zmq.PAIR)
        simulatorSocket.connect("tcp://localhost:%s" % simulatorPort)
        print("speedMeasure?> Connected to simulator %s" % simulatorPort)
    else:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FPS, 15)
    
    prevCx, prevCy = 0, 0
    prevTime = 0
    prevValid = False
    
    port = 5556
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)
    
    lastCycleTimestamp = time.time()
    while True:
        if not pause:
            if useCamera:
                flag = False
                pos_frame = 0
                while not flag:
                    flag, frame = cam.read()
                    actCycleTimestamp = time.time()
                    print("dTime", actCycleTimestamp-lastCycleTimestamp)
                    lastCycleTimestamp = actCycleTimestamp
                    if not flag:
                        pass
                #pos_frame = cam.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                frame = simulatorSocket.recv_pyobj()
                simulatorSocket.send_pyobj("OK")
                
            actTime = time.time()
            # Our operations on the frame come here
            if refreshAruco and not manualSpeed:
                aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                parameters =  aruco.DetectorParameters_create()
                corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
           
            if len(corners) > 0:
                # Draw it for debugging    
                frame = aruco.drawDetectedMarkers(frame, corners)
                floorCoords = np.float32([[aruco_dim, 0], [aruco_dim, aruco_dim], [0, aruco_dim], [0, 0]])
                screenCoords = corners[0][0] # I use the first found marker
                
                M = cv2.getPerspectiveTransform(floorCoords, screenCoords)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, thres = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY) # Usually, it's easier to work with binary images
                thres = cv2.dilate(thres, np.ones((3,3)), iterations = 2 )
                thres = cv2.erode(thres, np.ones((3,3)), iterations = 3 ) # noise filtering
                massCenter = getMassCenter(thres)        
                skeleton = getSkeleton(thres)
                
                if massCenter[0] is not None:
                    closestPoint = (0,0)
                    closestDistanceSquare = 99999999
                    
                    for p in getNonzeroPixels(skeleton):
                        distanceSquare = getDistanceSquare(p, massCenter)
                        if distanceSquare < closestDistanceSquare:
                            closestDistanceSquare = distanceSquare
                            closestPoint = p
                    
                    #cv2.circle(frame, massCenter, 1, (0,255,0), 10)
                    cv2.circle(frame, closestPoint, 1, (255,0,0), 10)
                    #cv2.circle(thres, closestPoint, 1, (127,), 10)
                    #cv2.circle(skeleton, closestPoint, 1, (127,), 10)
                    valid = True
                    
                    cx, cy = convertFromScreenToFloorCoordSystem(closestPoint[0], closestPoint[1], M)
                else:
                    valid = False
                    
                if valid and prevValid:
                    dx = cx - prevCx
                    dy = cy - prevCy
                    dTime = actTime-prevTime
                    distance = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))/10 # Ez csak skálázás
                    
                    data = {'distance':distance,'dTime':dTime, 'speed':distance, 'x':cx, 'y':cy} # distance / dTime / 10
                    if data['speed'] < 100:
                        # In the simulation, if the car leaves one side and enters an other, it causes a spike
                        #print data
                        socket.send_pyobj(data)
            
                if valid:
                    prevValid = True
                    prevCx, prevCy, prevTime = cx, cy, actTime
                    
                if calculateCircle:
                    circleBuffer.append((cx, cy))
                    if len(circleBuffer) > 1:
                        center, radius = calculateCircleData(np.array(circleBuffer))
                        cv2.line(frame, (int(center[0]), int(center[1])), (int(center[0])+int(radius)-10, int(center[1])), (127,))
                        cv2.putText(frame, str(int(radius))+" cm", (int(center[0]), int(center[1])+20), 1, 1, (127,))
                        cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (127,), 1)
                else:
                    circleBuffer = []
            else:
                print("No aruco code found")
            
        if manualSpeed:
            data = {'distance':0,'dTime':0, 'speed':manualSpeedValue, 'x':0, 'y':0}
            socket.send_pyobj(data)
            
        if show == 0:
            cv2.imshow('Main', frame)
        elif show == 1:
            cv2.imshow('Main', thres)
        elif show == 2:
            cv2.imshow('Main', skeleton)
        
        pressedKey = cv2.waitKey(1) & 0xFF  

        if pressedKey == ord('q'):
            break
        elif pressedKey == ord('s'):
            show = (show + 1) % 3
        elif pressedKey == ord('p'):
            pause = not pause
        elif pressedKey == ord('c'):
            calculateCircle = not calculateCircle
        elif pressedKey == ord('a'):
            refreshAruco = not refreshAruco
            print("Aruco refresh ON" if refreshAruco else "Aruco refresh off")
        elif pressedKey == ord('+'):
            manualSpeedValue += 0.1
            print("Manual speed:", manualSpeedValue)
        elif pressedKey == ord('-'):
            manualSpeedValue -= 0.1
            print("Manual speed:", manualSpeedValue)
            
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
