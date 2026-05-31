import numpy as np
import cv2
from inference import get_model
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration

class ViewTransformer:
    def __init__(self,source:np.ndarray,target:np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        
        if source.shape[1] != 2:
             raise ValueError("Source and target points must be 2D coordinates.")
         
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        self.m,_ = cv2.findHomography(srcPoints=source,dstPoints=target)
        
        if self.m is None:
            raise ValueError( "Homography matrix could not be calculated.")
        
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)
    

pitch_model = get_model(
    model_id="football-field-detection-f07vi/14",
    api_key="o4DIKHkmwkGnr5GAxvSL"
)

CONFIG = SoccerPitchConfiguration()


def get_tranfromation(frame):
    results = pitch_model.infer(frame, confidence=0.3)[0]
    keypoints = sv.KeyPoints.from_inference(results)
    
    filter = keypoints.confidence[0] > 0.5

    source = keypoints.xy[0][filter].astype(np.float32)
    target = np.array(CONFIG.vertices)[filter].astype(np.float32) / 100
    
    transformer = ViewTransformer(source=source, target=target)
    return transformer,source

