import cv2
import coremltools
from PIL import Image
import numpy as np
import time
from utils import *  # Assuming this includes necessary functions like plot_rectangles
from _tracker import CentroidTracker
from collections import defaultdict
from circular_buffer import CircularBuffer


# Constants
VIDEO_FILE = 'videos/output.mp4'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8
MODEL_PATH = 'models/last.mlpackage'
GRAB_REGION = (800, 0, 950, 250)
FORWARD_REGION = (166, 263, 513, 400)
BACKWARD_REGION = (1000, 300, 1300, 400)
MACHINE_REGION = (530, 270, 800, 400)
STATS_ZONE = (1500, 800, 1920, 1080)
stats = {'grab': 0,
         'forward': 0,
         'backward':0,
         'machine':0
         }
logger = []

forward_zone_tracker = CentroidTracker(maxDisappeared=300)
backward_zone_tracker = CentroidTracker(maxDisappeared=300)
grab_zone_tracker = CentroidTracker(maxDisappeared=300)
machine_zone_tracker = CentroidTracker(maxDisappeared=300)

forward_zone_history = defaultdict(lambda: [])
backward_zone_history = defaultdict(lambda: [])
grab_zone_history = defaultdict(lambda: [])
machine_zone_history = defaultdict(lambda: [])

tracking_history_grab = CircularBuffer(10)


# Load CoreML model

model = coremltools.models.MLModel(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_FILE)
# Advance the video by 1 minute and 15 seconds
frame_rate = cap.get(cv2.CAP_PROP_FPS)
#skip_time = 0*60+30  # 1 minute and 15 seconds in seconds
#skip_time = 8*60+30  # 1 minute and 15 seconds in seconds
skip_time = 0*60+0  # 1 minute and 15 seconds in seconds

skip_frames = int(frame_rate * skip_time)
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)



def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the window title with the current coordinates of the mouse
        cv2.setWindowTitle('Window', f'Coordinates: ({x}, {y})')
cv2.namedWindow("Window", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Window", 1920, 1080)
cv2.setMouseCallback('Window', mouse_callback)

while True:
    start = time.time()
    success, img = cap.read()
    cv2.putText(img, frame_to_hms(cap.get(cv2.CAP_PROP_POS_FRAMES), frame_rate), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
    if not success:
        break

    #extract blues
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 40, 40])  # Lower end of the blue spectrum
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    bluees = cv2.bitwise_and(img, img, mask=blue_mask)

    overlay_region(img, GRAB_REGION, alpha=0.5)
    overlay_region(img, FORWARD_REGION, alpha=0.5)
    overlay_region(img, BACKWARD_REGION, alpha=0.5)
    overlay_region(img, MACHINE_REGION, alpha=0.5)
    overlay_region(img, STATS_ZONE, alpha=1)

    forward_zone_rects = []
    backward_zone_rects = []
    grab_zone_rects = []
    machine_zone_rects = []


    # Preprocess the image
    processed_img = cv2.resize(img, (384, 224))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(processed_img)

    # Run inference
    mstart = time.time()
    results = model.predict({'image': pil_img, 
                             'iouThreshold': IOU_THRESHOLD, 
                             'confidenceThreshold': CONFIDENCE_THRESHOLD})
    mstop = time.time()
    # Extract the confidence scores and coordinates
    confidences = results['confidence']
    coordinates = results['coordinates']

    grab_appeared = 0
    # Draw bounding boxes
    for confidence, (xn, yn, widthn, heightn) in zip(confidences, coordinates):
        if confidence > CONFIDENCE_THRESHOLD:
            # Scale coordinates
            x = xn * img.shape[1]
            y = yn * img.shape[0]
            x1 = int((xn - widthn/2) * img.shape[1])
            y1 = int((yn - heightn/2) * img.shape[0])
            x2 = int((xn + widthn/2) * img.shape[1])
            y2 = int((yn + heightn/2) * img.shape[0])
            
            plot_rectangles1(img, x1,y1,x2,y2,confidence)

            #push to trackers
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),GRAB_REGION)):
                grab_appeared = 1
                grab_zone_rects.append([x1,y1,x2,y2])
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),FORWARD_REGION)):
                forward_zone_rects.append([x1,y1,x2,y2])
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),BACKWARD_REGION)):
                backward_zone_rects.append([x1,y1,x2,y2])
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),MACHINE_REGION)):
                machine_zone_rects.append([x1,y1,x2,y2])

            #update trackers
            grab_zone_objects = grab_zone_tracker.update(grab_zone_rects)
            print(grab_zone_objects)
            plot_path(img, grab_zone_objects, grab_zone_tracker, grab_zone_history)
            #stats['grab'] = grab_zone_tracker.count

            forward_zone_objects = forward_zone_tracker.update(forward_zone_rects)
            plot_path(img, forward_zone_objects, forward_zone_tracker, forward_zone_history)
            #stats['forward'] = forward_zone_tracker.count
            
            backward_zone_objects = backward_zone_tracker.update(backward_zone_rects)
            plot_path(img, backward_zone_objects, backward_zone_tracker, backward_zone_history)
            #stats['backward'] = backward_zone_tracker.count
            
            machine_zone_objects = machine_zone_tracker.update(machine_zone_rects)
            plot_path(img, machine_zone_objects, machine_zone_tracker, machine_zone_history)
            #stats['machine'] = machine_zone_tracker.count

    tracking_history_grab.append(grab_appeared)
    print(tracking_history_grab)
    if(tracking_history_grab.sum() > 5):
        stats['grab'] += 1 

    plot_stats(img, STATS_ZONE, stats)
    cv2.imshow('Window', img)
    #cv2.imshow('bluees', bluees)
    stop = time.time()
    inference = mstop - mstart
    total = stop - start
    print(f'total time:{total*1000} inference time: {inference*1000}')
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
