import cv2
import numpy as np
from math import floor, ceil
from PIL import Image


lower_blue = np.array([100, 40, 40])  # Lower end of the blue spectrum
upper_blue = np.array([140, 255, 255])
lower_white = np.array([0, 0, 200], dtype=np.uint8)
upper_white = np.array([255, 30, 255], dtype=np.uint8)

lower_red = np.array([0, 0, 200], dtype = "uint8")
upper_red= np.array([50, 10, 255], dtype = "uint8")

classNames = ["handle"]

def plot_stats(img, zone, stats, label):
    offset = 0
    cv2.putText(img, f'{label}', (zone[0], zone[1]+offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
    for k, v in list(stats.items())[:-1]:
        offset+=60
        cv2.putText(img, f'{k}:{v}', (zone[0], zone[1]+offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
def get_stats(machine_zone_tracker, grab_zone_tracker,r):
    handle_count = machine_zone_tracker.count/2
    bags = floor(handle_count/4)
    current_bag_progress = handle_count/4 - bags
    fps = int(1000 / (sum(r.speed.values())))
    handles_grabbed = grab_zone_tracker.count
    return handle_count, bags, current_bag_progress, handles_grabbed, fps

def plot_dashboard(img, current_bag_progress, handle_count, bags, handles_grabbed, fps):
    cv2.putText(img, f'cuurent bag progress: {current_bag_progress*100}%', [300, 175], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    cv2.putText(img, f'handles made: {floor(handle_count)}    |    handles grabbed: {handles_grabbed}    |    handles_thrown: {max(floor(handle_count - handles_grabbed), 0)}', [300, 250], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    cv2.rectangle(img, (300, 275), (750, 340), (0, 255, 255), -1)
    cv2.putText(img, f'bags made: {bags}', [300, 325], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
    cv2.putText(img, f'Dashboard', [1675, 80], cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    plot_progress_bar(current_bag_progress*100, img, (1240, 120, 3444, 195))
    cv2.putText(img, f'FPS: {fps}', [300, 400], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

def plot_rectangles(img, box, no_label=False):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = ceil((box.conf[0] * 100)) / 100
    cls = int(box.cls[0])
    if no_label:
        cv2.putText(img, f'', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
    else:
        cv2.putText(img, f'{classNames[cls]} {confidence}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 5)

def plot_rectangles1(img, x1,y1,x2,y2, confidence,no_label=False):
    if no_label:
        cv2.putText(img, f'', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
    else:
        cv2.putText(img, f'{confidence[0]:0.2f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 5)

def frame_to_hms(frame_number, frame_rate):
    # Calculate total seconds
    total_seconds = frame_number / frame_rate
    
    # Calculate hours, minutes and seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    # Format time as hh:mm:ss
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def plot_progress_bar(progress, img, coordinates):
    num_boxes = int(progress / 12.5)
    xmin, ymin, xmax, ymax = coordinates
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 8)
    box_width = int((xmax - xmin) / (8 + 7 * 0.05)) 
    box_spacing = int(0.05 * box_width)
    start = xmin + floor(box_spacing)
    for i in range(num_boxes):
        cv2.rectangle(img, (start, ymin + 10), (start + box_width, ymax -11), (0, 255, 255), -1)
        start = start + box_width + box_spacing
        
    text_x = int((xmin + xmax) / 2)
    text_y = int((ymin + ymax) / 2)

    progress_str = "{:.1f}%".format(num_boxes/8*100)
    if(progress > 40):
        cv2.putText(img, progress_str, (text_x - 100, text_y+15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
    else:
        cv2.putText(img, progress_str, (text_x - 100, text_y+15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)


def box_area(xyxy):
    return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) 

def centroid_in_zone(centroid, xyxy, zone):
    return zone[0] <= centroid[0] <= zone[2] and zone[1] <= centroid[1] <= zone[3]# and box_area(xyxy) > 25000

def overlay_region(img, region, alpha=0.1, color=(0, 0, 255)):
    machine_overlay = np.full_like(img[region[1]:region[3], region[0]:region[2]], (250, 216, 230), dtype=np.uint8)
    cv2.addWeighted(machine_overlay, alpha, img[region[1]:region[3], region[0]:region[2]], 0.5, 0, img[region[1]:region[3], region[0]:region[2]])
    cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), color, 3)

def plot_path(img, objects, zone, history):
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            if zone.disappeared[objectID] == 0:
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            track = history[objectID]
            track.append((centroid[0] - 10, centroid[1] - 10))
            if len(track) > 60:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points], isClosed=False, color=(255, 0, 255), thickness=5)  

def get_coordinates(img, xn, yn, widthn, heightn):
    x = xn * img.shape[1]
    y = yn * img.shape[0]
    x1 = int((xn - widthn/2) * img.shape[1])
    y1 = int((yn - heightn/2) * img.shape[0])
    x2 = int((xn + widthn/2) * img.shape[1])
    y2 = int((yn + heightn/2) * img.shape[0])
    return x, y, x1, y1, x2, y2
def plot_time_on_frame(img, cap, fps):
    cv2.putText(img, frame_to_hms(cap.get(cv2.CAP_PROP_POS_FRAMES), fps), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

def preprocess_img(img):
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_img)

def get_blue(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    return cv2.bitwise_and(img, img, mask=blue_mask)
def get_white(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    return cv2.bitwise_and(img, img, mask=white_mask)
def get_red(img):
    # Define the constant red color
    #target_red = np.array([30, 3, 214])

    #hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #print(hsv_image)

    red_mask = cv2.inRange(img, lower_red, upper_red)
    return cv2.bitwise_and(img, img, mask=red_mask)

def binary_image(img, threshold_value=180):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_result = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_result
    
def init_current_stats():
    return {'grab': 0,
         'forward': 0,
         'backward':0,
         'machine':0,
         'events': []
         }

def print_disappearing_message(frame, zone, message, current_frame, total_frames):
    # Calculate opacity based on the current frame
    opacity = int(255 * (1 - current_frame / total_frames))

    # Create an overlay image with the message
    overlay = frame.copy()
    cv2.putText(overlay, message, (zone[0], zone[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

    # Apply the opacity to the overlay
    cv2.addWeighted(overlay, opacity / 255.0, frame, 1 - opacity / 255.0, 0, frame)
def plot_logs(img, zone, logs):
    offset = 0
    for log in logs:
        print_disappearing_message(img, (zone[0], zone[1]+offset), log['message'], log['total_frames'] - log['frames_left'], log['total_frames'])
        log['frames_left'] -= 1
        offset = offset - 50

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the window title with the current coordinates of the mouse
        cv2.setWindowTitle('Window', f'Coordinates: ({x}, {y})')