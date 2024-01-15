import cv2
import coremltools
from PIL import Image
import numpy as np
import time
from utils import *  
from custom_tracker import CentroidTracker
from collections import defaultdict
from circular_buffer import CircularBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt

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
MACHINE_ZONE = (525, 270, 850, 450)
TABLE_ZONE = (525, 370, 970, 1075)

FORWARD_DIRECTIONS = [' left', 'up left', 'down left', 'up', 'down', ' ']
FORWARD_DIRECTIONS = [' left', 'up left', 'down left']

GRAB_TEMPRATURE = 3
FORWARD_TEMPRATURE = 7
BACKWARD_TEMPRATURE = 7
MACHINE_TEMPRATURE = 7

BLUE = True
##############################################

################INIT_TRACKER############################
grab_tracker = CentroidTracker(maxDisappeared=300, direction=[])
grab_history = defaultdict(lambda: [])
tracking_history_grab = CircularBuffer(20)
grab_id_dict = {}
##############################################
forward_tracker = CentroidTracker(maxDisappeared=300, direction=FORWARD_DIRECTIONS)
forward_history = defaultdict(lambda: [])
tracking_history_forward = CircularBuffer(40)
forward_id_dict = {}
##############################################
backward_tracker = CentroidTracker(maxDisappeared=150, direction=FORWARD_DIRECTIONS)
#TODO directoinal tracker doesnt work here
#behaviour when object goes into oposit direction and comes back it doesn't get registred
#as new one and rather keeps gettting ignored by tracker
backward_history = defaultdict(lambda: [])
tracking_history_backward = CircularBuffer(40)
backward_id_dict = {}
##############################################
machine_tracker = CentroidTracker(maxDisappeared=150, direction=FORWARD_DIRECTIONS)
machine_history = defaultdict(lambda: [])
tracking_history_machine = CircularBuffer(40)
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
#skip_time = 4*60+0
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
skip_time = 5*60+50
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

table_list = []
lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-10)
#lenght = 7*60+7
#lenght = int(frame_rate * lenght)
print(f'len = {lenght}')
#while True:
for _ in tqdm(range(lenght)):
    #if cap.get(cv2.CAP_PROP_POS_FRAMES) > frame_rate * skip_time+ 38*frame_rate:
    #    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_rate * skip_time+9))
    start = time.time()
    success, img = cap.read()
    plot_time_on_frame(img, cap, frame_rate)

    grab_rects = []
    forward_rects = []
    backward_rects = []
    machine_rects = []

    grab_appeared_flag = 0
    forward_appeared_flag = 0
    backward_appeared_flag = 0
    machine_appeared_flag = 0
    flagXgrab = 0
    flagXforward = 0
    flagXbackward = 0
    flagXmachine = 0
    resized_img = cv2.resize(img, (384, 224))

    y1 = int((TABLE_ZONE[1]/img.shape[0])*resized_img.shape[0])
    y2 = int((TABLE_ZONE[3]/img.shape[0])*resized_img.shape[0])
    x1 = int((TABLE_ZONE[0]/img.shape[1])*resized_img.shape[1])
    x2 = int((TABLE_ZONE[2]/img.shape[1])*resized_img.shape[1])

    #white = get_white(resized_img[y1:y2,x1:x2])
    #binarized = binary_image(white)
    #binarized2 = binary_image(resized_img[y1:y2,x1:x2])
    #table_list.append(np.sum(binarized2))
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
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),GRAB_ZONE)):
                grab_appeared_flag = 1
                grab_rects = [int(x), int(y)]

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),FORWARD_ZONE)):
                forward_appeared_flag = 1
                forward_rects = [int(x), int(y)]

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),BACKWARD_ZONE)):
                backward_appeared_flag = 1
                backward_rects = [int(x), int(y)]

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),MACHINE_ZONE)):
                machine_appeared_flag = 1
                machine_rects = [int(x), int(y)]

   
    grab_objects, flagXgrab = grab_tracker.update(grab_rects)
    plot_path(img, grab_objects, grab_tracker, grab_history)
    if(str(grab_tracker.count) not in grab_id_dict.keys()):
        grab_id_dict[str(grab_tracker.count)] = 0
    if(grab_appeared_flag == 1 and grab_id_dict[str(grab_tracker.count)] == 0):
        tracking_history_grab.append(flagXgrab)
    else:
        tracking_history_grab.append(0)

    if(flagXgrab):
        if(grab_id_dict[str(grab_tracker.count)] == 0):
            if(tracking_history_grab.sum() > GRAB_TEMPRATURE):
                grab_id_dict[str(grab_tracker.count)] = 1
                tracking_history_grab.flush()
                stats['grab'] += 1          

    forward_objects, flagXforward = forward_tracker.update(forward_rects)
    plot_path(img, forward_objects, forward_tracker, forward_history)
    if(str(forward_tracker.count) not in forward_id_dict.keys()):
        forward_id_dict[str(forward_tracker.count)] = 0
    if(forward_appeared_flag == 1 and forward_id_dict[str(forward_tracker.count)] == 0):
        tracking_history_forward.append(flagXforward)
    else:
        tracking_history_forward.append(0)

    if(flagXforward):
        if(forward_id_dict[str(forward_tracker.count)] == 0):
            if(tracking_history_forward.sum() > FORWARD_TEMPRATURE):
                forward_id_dict[str(forward_tracker.count)] = 1
                tracking_history_forward.flush()
                stats['forward'] += 1 

    backward_objects, flagXbackward = backward_tracker.update(backward_rects)
    plot_path(img, backward_objects, backward_tracker, backward_history)
    if(str(backward_tracker.count) not in backward_id_dict.keys()):
        backward_id_dict[str(backward_tracker.count)] = 0
    if(backward_appeared_flag == 1 and backward_id_dict[str(backward_tracker.count)] == 0):
        tracking_history_backward.append(flagXbackward)
    else:
        tracking_history_backward.append(0)

    if(flagXbackward):
        if(backward_id_dict[str(backward_tracker.count)] == 0):
            if(tracking_history_backward.sum() > BACKWARD_TEMPRATURE):
                backward_id_dict[str(backward_tracker.count)] = 1
                tracking_history_backward.flush()
                stats['backward'] += 1 

    machine_objects, flagXmachine = machine_tracker.update(machine_rects)
    plot_path(img, machine_objects, machine_tracker, machine_history)
    if(str(machine_tracker.count) not in machine_id_dict.keys()):
        machine_id_dict[str(machine_tracker.count)] = 0
    if(machine_appeared_flag == 1 and machine_id_dict[str(machine_tracker.count)] == 0):
        tracking_history_machine.append(flagXmachine)
    else:
        tracking_history_machine.append(0)

    if(flagXmachine):
        if(machine_id_dict[str(machine_tracker.count)] == 0):
            if(tracking_history_machine.sum() > MACHINE_TEMPRATURE):
                machine_id_dict[str(machine_tracker.count)] = 1
                tracking_history_machine.flush()
                stats['machine'] += 1 



    overlay_region(img, GRAB_ZONE, alpha=0.5)
    overlay_region(img, FORWARD_ZONE, alpha=0.5)
    overlay_region(img, BACKWARD_ZONE, alpha=0.5)
    overlay_region(img, MACHINE_ZONE, alpha=0.5)
    #overlay_region(img, TABLE_ZONE, alpha=0.5)
    overlay_region(img, STATS_ZONE, alpha=1)
    plot_stats(img, STATS_ZONE, stats)
    #cv2.imshow('white', white)
    #cv2.imshow('binary', binarized)
    
    #cv2.imshow('binary2', binarized2)
    cv2.imshow('Window', img)

    stop = time.time()
    inference = mstop - mstart
    total = stop - start
    print(f'total time:{total*1000} inference time: {inference*1000}')
    print(stats)
    print(20*'-')
    # Show the plot
    if cv2.waitKey(1) == ord('q'):
        break
    
#np.save('binary_whites.npy', np.array(table_list))
#plt.plot(table_list)

#plt.show()

cap.release()
cv2.destroyAllWindows()
