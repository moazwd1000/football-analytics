import math
import numpy as np
import cv2
from homography import ViewTransformer


def get_center(box):
    x1,x2 = box[0],box[2]
    x_center = (x1+x2)/2
    y_down = box[3]
    return x_center,y_down


def calculate_speed(box1, box2, fps, transformer):
    
    x_center1,y_center1 = get_center(box1)
    x_center2,y_center2 = get_center(box2)
    
    real1 = transformer.transform_points(np.array([[x_center1, y_center1]]))[0]
    real2 = transformer.transform_points(np.array([[x_center2, y_center2]]))[0]
    
    distance = math.sqrt((real2[0]-real1[0])**2 + (real2[1]-real1[1])**2)    
    
    speed_ms = distance / (1/fps)
    return speed_ms * 3.6
    

def update_speed_history(track_id, speed, speed_history):
    if track_id not in speed_history:
        speed_history[track_id] = []
    speed_history[track_id].append(speed)
    speed_history[track_id] = speed_history[track_id][-10:]
    return speed_history


def get_smoothed_speed(track_id, speed_history):
    if track_id not in speed_history:
        return 0
    return sum(speed_history[track_id]) / len(speed_history[track_id])