import cv2
import numpy as np
from homography import get_tranfromation
from speed import get_center


def get_pitch_mask(frame, sam_model, points):
    result = sam_model.predict(frame, bboxes=points)
    return np.squeeze(result[0].masks.data[0].cpu().numpy().astype(np.uint8))


def filter_detections_by_mask(detections, mask, frame):
    filtered_dict = {}
    
    if frame.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    for cat in detections.keys():
        filtered_dict[cat] = []
        for detection in detections[cat]:
            x_center, y_down = get_center(detection["box"])
            x_center, y_down = int(x_center), int(y_down)
            x_center = np.clip(x_center, 0, mask.shape[1] - 1)
            y_down = np.clip(y_down, 0, mask.shape[0] - 1)

            if mask[y_down, x_center] == 1:
                filtered_dict[cat].append(detection)
        
    return filtered_dict
        

            
    





