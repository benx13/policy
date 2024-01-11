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
stats = {'zone': 0,
         'forward': 0,
         'backward':0,
         'machine':0
         }
logger = []
##############################################

################INIT_TRACKER############################
ZONE = (800, 0, 950, 250)
zone_tracker = CentroidTracker(maxDisappeared=300)#, direction=['left', 'up left', 'down left'])
zone_history = defaultdict(lambda: [])
tracking_history_zone = CircularBuffer(20)
id_dict = {}
##############################################

################INIT_MODEL############################
MODEL_PATH = 'models/last.mlpackage'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8
model = coremltools.models.MLModel(MODEL_PATH)
##############################################

################INIT_CAP############################
VIDEO_FILE = 'videos/output.mp4'
cap = cv2.VideoCapture(VIDEO_FILE)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
#skip_time = 0*60+30
#skip_time = 8*60+30
skip_time = 0*60+0
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
    overlay_region(img, ZONE, alpha=0.5)
    overlay_region(img, STATS_ZONE, alpha=1)

    #zone flags
    appeared = 0
    zone_rects = []

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

            #push to trackers
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),ZONE)):
                appeared = 1
                zone_rects.append([x1,y1,x2,y2])
                #update trackers
                zone_objects = zone_tracker.update(zone_rects)
                #print(zone_objects)
                plot_path(img, zone_objects, zone_tracker, zone_history)
                #stats['zone'] = zone_tracker.count
                if(str(zone_tracker.count) not in id_dict.keys()):
                    id_dict[str(zone_tracker.count)] = 0

                tracking_history_zone.append(zone_tracker.flag)
                print(tracking_history_zone)
                if(id_dict[str(zone_tracker.count)] == 0):
                    if(tracking_history_zone.sum() > 3):
                        id_dict[str(zone_tracker.count)] = 1
                        stats['zone'] += 1 
            else:
                tracking_history_zone.append(0)


    plot_stats(img, STATS_ZONE, stats)
    cv2.imshow('Window', img)
    stop = time.time()
    inference = mstop - mstart
    total = stop - start
    print(f'total time:{total*1000} inference time: {inference*1000}')
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
