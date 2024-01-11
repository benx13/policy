import cv2
import coremltools
from PIL import Image
import numpy as np
import time
from utils import *  
from tracker import CentroidTracker
from collections import defaultdict
from circular_buffer import CircularBuffer


###################INIT_STATS###########################
STATS_ZONE = (1500, 800, 1920, 1080)
stats = {'grab': 0,
         'forward': 0,
         'backward':0,
         'machine':0
         }
logger = []
##############################################

###################HYPER_PARAMS###########################
GRAB_ZONE = (750, 0, 1000, 270)
FORWARD_ZONE = (166, 263, 560, 450)
BACKWARD_ZONE = (970, 270, 1300, 450)
MACHINE_ZONE = (560, 270, 800, 450)

FORWARD_DIRECTIONS = ['left', 'up left', 'down left', 'up', 'down']

GRAB_TEMPRATURE = 3
FORWARD_TEMPRATURE = 3
BACKWARD_TEMPRATURE = 3
MACHINE_TEMPRATURE = 6

BLUE = True
##############################################

################INIT_TRACKER############################
grab_tracker = CentroidTracker(maxDisappeared=300)
grab_history = defaultdict(lambda: [])
tracking_history_grab = CircularBuffer(20)
grab_id_dict = {}
##############################################
forward_tracker = CentroidTracker(maxDisappeared=300, direction=FORWARD_DIRECTIONS)
forward_history = defaultdict(lambda: [])
tracking_history_forward = CircularBuffer(20)
forward_id_dict = {}
##############################################
backward_tracker = CentroidTracker(maxDisappeared=300)#, direction=['left', 'up left', 'down left', 'up', 'down'])
#TODO directoinal tracker doesnt work here
#behaviour when object goes into oposit direction and comes back it doesn't get registred
#as new one and rather keeps gettting ignored by tracker
backward_history = defaultdict(lambda: [])
tracking_history_backward = CircularBuffer(20)
backward_id_dict = {}
##############################################
machine_tracker = CentroidTracker(maxDisappeared=300)
machine_history = defaultdict(lambda: [])
tracking_history_machine = CircularBuffer(20)
machine_id_dict = {}
##############################################

################INIT_MODEL############################
MODEL_PATH = 'models/bluenano.mlpackage'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8
model = coremltools.models.MLModel(MODEL_PATH)
##############################################

################INIT_CAP############################
VIDEO_FILE = 'videos/IMG_2114.MOV'
cap = cv2.VideoCapture(VIDEO_FILE)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
skip_time = 0*60+0
#skip_time = 8*60+30
#skip_time = 0*60+22
#skip_time = 4*60+36
'''
Vid checkpoints
Bag1 : 0
Bag2: 8:33
Bag3: 16:15
Bag4:26 
Bag5:34
Bag6:40
Bag7:52:30
Bag8:59:15
Bag9::1:05:55
Bag10: 1:14:20
Bag11:1:20:00

------
tests:
bag1:   changed machine temp from 3 to 5
        anomaly at 4:36 caused by model false postive
        can be fixed by blue filter
        anomaly at 5:53 4th handle going backward fix by
        tuning zone and adding directions
'''
skip_frames = int(frame_rate * skip_time)
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
##############################################

###################INIT_IMSHOW###########################
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the window title with the current coordinates of the mouse
        cv2.setWindowTitle('Window', f'Coordinates: ({x}, {y})')
cv2.namedWindow("Window", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Window", 1920, 1080)
cv2.setMouseCallback('Window', mouse_callback)
########################################################



while True:
    start = time.time()
    success, img = cap.read()
    plot_time_on_frame(img, cap, frame_rate)
 

    grab_rects = []
    forward_rects = []
    backward_rects = []
    machine_rects = []
    if BLUE:
        blues = get_blue(img)
        input = preprocess_img(blues)
    else:
        input = preprocess_img(img)
    mstart = time.time()
    results = model.predict({'image': input, 
                             'iouThreshold': IOU_THRESHOLD, 
                             'confidenceThreshold': CONFIDENCE_THRESHOLD})
    mstop = time.time()

    for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
        if confidence > CONFIDENCE_THRESHOLD:
            x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
            plot_rectangles1(img, x1,y1,x2,y2,confidence)

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),GRAB_ZONE)):
                grab_rects.append([x1,y1,x2,y2])
            grab_objects = grab_tracker.update(grab_rects)
            plot_path(img, grab_objects, grab_tracker, grab_history)
            if(str(grab_tracker.count) not in grab_id_dict.keys()):
                grab_id_dict[str(grab_tracker.count)] = 0

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),FORWARD_ZONE)):
                forward_rects.append([x1,y1,x2,y2])
            forward_objects = forward_tracker.update(forward_rects)
            plot_path(img, forward_objects, forward_tracker, forward_history)
            if(str(forward_tracker.count) not in forward_id_dict.keys()):
                forward_id_dict[str(forward_tracker.count)] = 0

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),BACKWARD_ZONE)):
                backward_rects.append([x1,y1,x2,y2])
            backward_objects = backward_tracker.update(backward_rects)
            plot_path(img, backward_objects, backward_tracker, backward_history)
            if(str(backward_tracker.count) not in backward_id_dict.keys()):
                backward_id_dict[str(backward_tracker.count)] = 0

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),MACHINE_ZONE)):
                machine_rects.append([x1,y1,x2,y2])
            machine_objects = machine_tracker.update(machine_rects)
            plot_path(img, machine_objects, machine_tracker, machine_history)
            if(str(machine_tracker.count) not in machine_id_dict.keys()):
                machine_id_dict[str(machine_tracker.count)] = 0

    
    tracking_history_grab.append(grab_tracker.flag)
    #print(tracking_history_grab)
    #print(grab_id_dict)
    if(grab_tracker.flag):
        if(grab_id_dict[str(grab_tracker.count)] == 0):
            if(tracking_history_grab.sum() > GRAB_TEMPRATURE):
                grab_id_dict[str(grab_tracker.count)] = 1
                stats['grab'] += 1 

    tracking_history_forward.append(forward_tracker.flag)
    #print(tracking_history_forward)
    #print(forward_id_dict)
    if(forward_tracker.flag):
        if(forward_id_dict[str(forward_tracker.count)] == 0):
            if(tracking_history_forward.sum() > FORWARD_TEMPRATURE):
                forward_id_dict[str(forward_tracker.count)] = 1
                stats['forward'] += 1 

    tracking_history_backward.append(backward_tracker.flag)
    #print(tracking_history_backward)
    #print(backward_id_dict)
    if(backward_tracker.flag):
        if(backward_id_dict[str(backward_tracker.count)] == 0):
            if(tracking_history_backward.sum() > BACKWARD_TEMPRATURE):
                backward_id_dict[str(backward_tracker.count)] = 1
                stats['backward'] += 1 

    tracking_history_machine.append(machine_tracker.flag)
    print(tracking_history_machine)
    print(machine_id_dict)
    if(machine_tracker.flag):
        if(machine_id_dict[str(machine_tracker.count)] == 0):
            if(tracking_history_machine.sum() > MACHINE_TEMPRATURE):
                machine_id_dict[str(machine_tracker.count)] = 1
                stats['machine'] += 1 

    overlay_region(img, GRAB_ZONE, alpha=0.5)
    overlay_region(img, FORWARD_ZONE, alpha=0.5)
    overlay_region(img, BACKWARD_ZONE, alpha=0.5)
    overlay_region(img, MACHINE_ZONE, alpha=0.5)
    overlay_region(img, STATS_ZONE, alpha=1)
    plot_stats(img, STATS_ZONE, stats)
    cv2.imshow('Window', img)
    stop = time.time()
    inference = mstop - mstart
    total = stop - start
    print(f'total time:{total*1000} inference time: {inference*1000}')
    print(stats)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
