import cv2
import coremltools
from PIL import Image
import numpy as np
import time
from utils import *  
from custom_tracker import CentroidTracker
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

MACHINE_ZONE = (750, 0, 1030, 275)
MACHINE_ZONE_2 = (975, 215, 1100, 325)
FORWARD_DIRECTIONS = [' right', ' ']#, 'up right', 'down right']#, 'up left', 'down left']#, 'up left', 'down left', 'up', 'down', ' ']

MACHINE_TEMPRATURE = 10

BLUE = True
##############################################

################INIT_TRACKER############################
machine_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)#, direction=FORWARD_DIRECTIONS)
machine_history = defaultdict(lambda: [])
tracking_history_machine = CircularBuffer(100)
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
#skip_time = 3*60+50
#skip_time = 8*60+30
skip_time = 76*60+0
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
    #if cap.get(cv2.CAP_PROP_POS_FRAMES) > frame_rate * skip_time+ 38*frame_rate:
    #    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_rate * skip_time+9))
    start = time.time()
    success, img = cap.read()
    plot_time_on_frame(img, cap, frame_rate)
 

    machine_rects = []

    machine_appeared_flag = 0
    flagX = 0
    resized_img = cv2.resize(img, (384, 224))

    if BLUE:
        blues = get_blue(resized_img)
        input = preprocess_img(blues)
    else:
        input = preprocess_img(resized_img)
    mstart = time.time()
    results = model.predict({'image': input, 
                             'iouThreshold': IOU_THRESHOLD, 
                             'confidenceThreshold': CONFIDENCE_THRESHOLD})
    mstop = time.time()

    for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
        if confidence > CONFIDENCE_THRESHOLD:
            x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
            plot_rectangles1(img, x1,y1,x2,y2,confidence)

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),MACHINE_ZONE) or centroid_in_zone((x, y), (x1, y1, x2, y2),MACHINE_ZONE_2)):
                print('object detected')
                machine_appeared_flag = 1
                machine_rects = [int(x), int(y)]
            



    machine_objects, flagX = machine_tracker.update(machine_rects)
    plot_path(img, machine_objects, machine_tracker, machine_history)
    if(str(machine_tracker.count) not in machine_id_dict.keys()):
        machine_id_dict[str(machine_tracker.count)] = 0
    if(machine_appeared_flag == 1 and machine_id_dict[str(machine_tracker.count)] == 0):
        tracking_history_machine.append(flagX)
    else:
        tracking_history_machine.append(0)

    if(flagX):
        if(machine_id_dict[str(machine_tracker.count)] == 0):
            if(tracking_history_machine.sum() > MACHINE_TEMPRATURE):
                machine_id_dict[str(machine_tracker.count)] = 1
                tracking_history_machine.flush()
                stats['machine'] += 1 


    print(f'tracking_history_machine --> {tracking_history_machine}')
    #print(f'machine_id_dict --> {machine_id_dict}')
    #print(f'machine_appeared_flag --> {machine_appeared_flag}')
    #print(f'machine_id_dict[str(machine_tracker.count)] --> {machine_id_dict[str(machine_tracker.count)]}')
    #print(f'flag in main loop -->{flagX}')


    overlay_region(img, MACHINE_ZONE, alpha=0.5)
    overlay_region(img, MACHINE_ZONE_2, alpha=0.5)
    overlay_region(img, STATS_ZONE, alpha=1)
    plot_stats(img, STATS_ZONE, stats)
    cv2.imshow('Window', img)
    stop = time.time()
    inference = mstop - mstart
    total = stop - start
    #print(f'total time:{total*1000} inference time: {inference*1000}')
    print(stats)
    print(20*'-')
    if cv2.waitKey(100) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
