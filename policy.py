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
from counter import Counter
from logger import Logger
###################INIT_STATS###########################
TOTAL_STATS_ZONE = (1500, 750, 1920, 1080)
CURRENT_STATS_ZONE = (1500, 450, 1920, 700)
logger = Logger()
##############################################

###################HYPER_PARAMS###########################
GRAB_ZONE = (750, 0, 1030, 275)
GRAB_ZONE_2 = (975, 215, 1100, 325)
FORWARD_ZONE = (166, 263, 560, 450)
BACKWARD_ZONE = (970, 270, 1300, 450)
MACHINE_ZONE = (525, 270, 850, 450)
TABLE_ZONE = (525, 370, 970, 1075)
RED_ZONE = (1870, 95, 1885, 110)

FORWARD_DIRECTIONS = [' left', 'up left', 'down left', 'up', 'down', ' ']
BACKWARD_DIRECTIONS = [' right', ' ']

GRAB_TEMPRATURE = 3
FORWARD_TEMPRATURE = 7
BACKWARD_TEMPRATURE = 5
MACHINE_TEMPRATURE = 10
TRANSITION_TEMPRATURE = 40

BLUE = True

################INIT_TRACKER############################
grab_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)
grab_history = defaultdict(lambda: [])
grab_counter = Counter(grab_tracker, GRAB_TEMPRATURE, 100)
############################################
transition_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)
transition_history = defaultdict(lambda: [])
transition_counter = Counter(transition_tracker, TRANSITION_TEMPRATURE, 100)
##############################################
forward_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200, direction=[' left', 'up left', 'down left'])
forward_history = defaultdict(lambda: [])
forward_counter = Counter(forward_tracker, FORWARD_TEMPRATURE, 100)
##############################################
backward_tracker = CentroidTracker(maxDisappeared=2000, minDistanece=200, direction=BACKWARD_DIRECTIONS)
backward_history = defaultdict(lambda: [])
backward_counter = Counter(backward_tracker, BACKWARD_TEMPRATURE, 100)
##############################################
machine_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200, direction=[' left', 'up left', 'down left', ' '])
machine_history = defaultdict(lambda: [])
machine_counter = Counter(machine_tracker, MACHINE_TEMPRATURE, 100)
##############################################

################INIT_MODEL############################
MODEL_PATH = 'models/bluenano.mlpackage'
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.8
model = coremltools.models.MLModel(MODEL_PATH)
##############################################

################INIT_CAP############################
VIDEO_FILE = 'videos/test2.mov'
cap = cv2.VideoCapture(VIDEO_FILE)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
skip_time = 34*60+0
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
bags = 0
lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-10
print(f'len = {lenght}')
for _ in tqdm(range(lenght)):
    start = time.time()
    success, img = cap.read()
    plot_time_on_frame(img, cap, frame_rate)

    transition_counter.reset()
    grab_counter.reset()
    forward_counter.reset()
    backward_counter.reset()
    machine_counter.reset()

    resized_img = cv2.resize(img, (384, 224))

    reds = img[RED_ZONE[1]:RED_ZONE[3], RED_ZONE[0]:RED_ZONE[2]]
    reds2 = get_red(reds)

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

    if(np.sum(reds2[:,:,2])>0):
            grab_counter.update([1880, 100])
    
    for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
        if confidence > CONFIDENCE_THRESHOLD:
            x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
            plot_rectangles1(img, x1,y1,x2,y2,confidence)
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),GRAB_ZONE) or centroid_in_zone((x, y), (x1, y1, x2, y2),GRAB_ZONE_2)):
                grab_counter.update([int(x), int(y)])

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),FORWARD_ZONE)):
                forward_counter.update([int(x), int(y)])

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),BACKWARD_ZONE)):
                backward_counter.update([int(x), int(y)])

            if(centroid_in_zone((x, y), (x1, y1, x2, y2),MACHINE_ZONE)):
                machine_counter.update([int(x), int(y)])
    current_time = frame_to_hms(cap.get(cv2.CAP_PROP_POS_FRAMES), frame_rate)
    #---------------
    flagXtransition = transition_counter.apply()
    if(flagXtransition):
        logger.update('transition', current_time)
    #---------------
    flagXgrab = grab_counter.apply()
    plot_path(img, grab_tracker.objects, grab_tracker, grab_history)
    if(flagXgrab):
        logger.update('grab', current_time)
    #---------------
    flagXforward = forward_counter.apply()
    plot_path(img, forward_tracker.objects, forward_tracker, forward_history)
    if(flagXforward):
        logger.update('forward', current_time)
    #---------------
    flagXbackward = backward_counter.apply()
    plot_path(img, backward_tracker.objects, backward_tracker, backward_history)
    if(flagXbackward):
        logger.update('backward', current_time)
    #---------------
    flagXmachine = machine_counter.apply()
    plot_path(img, machine_tracker.objects, machine_tracker, machine_history)
    if(flagXmachine):
        logger.update('machine', current_time)
    #---------------



    overlay_region(img, GRAB_ZONE, alpha=0.5)
    overlay_region(img, GRAB_ZONE_2, alpha=0.5)
    overlay_region(img, FORWARD_ZONE, alpha=0.5)
    overlay_region(img, BACKWARD_ZONE, alpha=0.5)
    overlay_region(img, MACHINE_ZONE, alpha=0.5)
    #overlay_region(img, TABLE_ZONE, alpha=0.5)
    overlay_region(img, TOTAL_STATS_ZONE, alpha=1)
    overlay_region(img, CURRENT_STATS_ZONE, alpha=1)
    #------------------------
    plot_stats(img, TOTAL_STATS_ZONE, logger.stats['total'], 'total_stats')
    plot_stats(img, CURRENT_STATS_ZONE, logger.buffer, 'current_stats')
    #---------------------
    logger.update_logs()
    plot_logs(img, (0, 1060), logger.logs)
    #-----------------
    cv2.imshow('Window', img)
    #------------------
    stop = time.time()
    inference = mstop - mstart
    total = stop - start
    #------------------
    print(f'total time:{total*1000} inference time: {inference*1000}')
    print(logger.stats)
    print(20*'-')
    #------------------
    if cv2.waitKey(1) == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
logger.save_results()
