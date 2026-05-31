from ultralytics import YOLO,SAM
import numpy as np
import cv2
from collections import defaultdict
from detection import detect_and_track
from speed import calculate_speed,update_speed_history,get_smoothed_speed
from homography import get_tranfromation
from heatmaps import generate_heatmap,update_positions,generate_team_heatmaps
from team_detections import TeamAssigner
from possession import get_possession,update_possession
from draw import draw_player,draw_ball,draw_referee,draw_goalkeeper,draw_possession,draw_pass_stats
from passes import _get_ball_holder, detect_pass, get_pass_stats,update_candidate
from segmentation import get_pitch_mask,filter_detections_by_mask

detection_model = YOLO("D:/Projects/football-analytics/models/best2.pt")
segmentation_model = SAM("D:/Projects/football-analytics/models/mobile_sam.pt")

previous_positions = {}
speed_history = {}
all_positions = defaultdict(list)
team_assignments     = {}
pass_events = []
team_ball_control = []
candidate = None
prev_holder = None
candidate_frames = 0
frame_idx = 0

team_assigner = TeamAssigner()


cap = cv2.VideoCapture("D:/Projects/football-analytics/data/raw/test - Trim.mp4")
fps, w, h = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

ret, first_frame = cap.read()
first_detections = detect_and_track(model=detection_model, frame=first_frame)
team_assigner.assign_team_color(first_frame, first_detections["players"])
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


_,source = get_tranfromation(first_frame)
source = [min(source[:,0]),min(source[:,1]),max(source[:,0]),max(source[:,1])]
print(source)
mask = get_pitch_mask(first_frame, segmentation_model, source)

while True:
    frame_idx = frame_idx + 1
    ret,frame = cap.read()
    
    if(frame_idx % fps ==0):
        _,source = get_tranfromation(frame)
        source = [min(source[:,0]),min(source[:,1]),max(source[:,0]),max(source[:,1])]
        mask = get_pitch_mask(frame, segmentation_model, source)
    
    if not ret:
        break
    detections = detect_and_track(model=detection_model,frame=frame)
    filtered_detections = filter_detections_by_mask(detections, mask, frame)
    players = filtered_detections["players"]
    goalkeepers = filtered_detections["goalkeepers"]
    referees = filtered_detections["referees"]
    balls = filtered_detections["balls"]
    
    try:
        transformer,_ = get_tranfromation(frame)
    except Exception as e:
        print(f"Homography failed: {e}")
        continue
            
    
    for player in players:
        track_id = player["id"]
        
        team_id =  team_assigner.get_player_team(frame, player["box"], track_id)
        player["team_id"] = team_id 
        team_assignments[player["id"]] = team_id
        
        if (team_id == 1):
            team_color = (255, 0, 0)
        else:
            team_color = (0, 255, 255)
        
        all_positions = update_positions(player, transformer, all_positions,track_id)
        
        speed = calculate_speed(previous_positions[player["id"]], player["box"], fps, transformer) if player["id"] in previous_positions else 0
        previous_positions[track_id] = player["box"]
        speed_history = update_speed_history(track_id, speed, speed_history)
        speed = get_smoothed_speed(track_id, speed_history)
        
        draw_player(frame, player, team_color, speed)


        
    possession = get_possession(balls, players, transformer)
    team_ball_control = update_possession(possession, team_ball_control)
    draw_possession(frame, possession, w)
        
    
     
    current_holder = _get_ball_holder(balls, players, transformer)
    
    candidate, candidate_frames, prev_holder, pass_events = update_candidate(current_holder, candidate,
                                                                             candidate_frames, prev_holder, pass_events, frame_idx)
    stats = get_pass_stats(pass_events)
    draw_pass_stats(frame, stats)
    
    for ball in balls:
        draw_ball(frame, ball)

    for gk in goalkeepers:
        draw_goalkeeper(frame, gk)

    for ref in referees:
        draw_referee(frame, ref)


    out.write(frame)
    

cap.release()
out.release()
heatmap_img1,heatmap_img2 = generate_team_heatmaps(all_positions, team_assignments)
cv2.imwrite("heatmap1.jpg", heatmap_img1)
cv2.imwrite("heatmap2.jpg", heatmap_img2)
stats = get_pass_stats(pass_events)
