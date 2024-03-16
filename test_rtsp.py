import cv2, time
import numpy as np
from utils import *

cap = cv2.VideoCapture("rtsp://192.168.0.100:5543/live/channel0")
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(frame_rate)

while(True):
  ret, frame = cap.read()

  if ret == 1:
    #frame = cv2.resize(frame, (640, 640))
    cv2.imshow('frame',frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()


