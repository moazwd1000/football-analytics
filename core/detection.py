from ultralytics import YOLO
import cv2

PLAYER = 3
BALL = 0
GOALKEEPER = 1
REFEREE = 2

def detect_and_track(frame, model, confidence_score=0.4):
    results = model.track(frame, tracker="bytetrack.yaml", persist=True)
    results = results[0]
    
    players = []
    goalkeepers = []
    referees = []
    balls = []
    
    for box in results.boxes:
        if box.id is None:
            continue
            
        cls = int(box.cls)
        conf = float(box.conf)
        track_id = int(box.id)
        xyxy = box.xyxy[0].cpu().numpy()
        
        detection = {
            "id": track_id,
            "box": xyxy,
            "confidence": conf
        }
        
        if cls == PLAYER and conf >= confidence_score:
            players.append(detection)
        elif cls == GOALKEEPER and conf >= confidence_score:
            goalkeepers.append(detection)
        elif cls == REFEREE and conf >= confidence_score:
            referees.append(detection)
        elif cls == BALL and conf >= 0.1:  # lower threshold for ball
            balls.append(detection)
    
    return {
        "players": players,
        "goalkeepers": goalkeepers,
        "referees": referees,
        "balls": balls
    }