from speed import get_center
import numpy as np
from math import sqrt
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def _get_ball_holder(balls, players, transformer):
    if len(balls) == 0:
        return None
    
    ball_position = balls[0]["box"]
    ball_x,ball_y = get_center(ball_position)
    ball_real = transformer.transform_points(np.array([[ball_x, ball_y]]))[0]

    min_distance = float('inf')
    closest_player = None    

    for player in players:
        player_position = player["box"]
        player_x,player_y = get_center(player_position)
        player_real = transformer.transform_points(np.array([[player_x, player_y]]))[0]
        
        distance = sqrt((player_real[0]-ball_real[0])**2 + (player_real[1]-ball_real[1])**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_player = player
            
    if min_distance<1.5:
        return closest_player
        
        

def detect_pass(prev_holder, curr_holder, frame_idx):
    if prev_holder is None:
        return None
    if curr_holder is None:
        return None
    
    if prev_holder != curr_holder:
        return {"from_player" : prev_holder["id"], "to_player" : curr_holder["id"],
                "from_team" : prev_holder["team_id"], "to_team":curr_holder["team_id"], "frame":frame_idx}


def get_pass_stats(pass_events):
    stats = {
        1: {"total": 0, "successful": 0},
        2: {"total": 0, "successful": 0}
    }
    
    for pass_event in pass_events:
        from_team = pass_event["from_team"]
        to_team = pass_event["to_team"]
        
        stats[from_team]["total"] += 1
        
        if from_team == to_team:
            stats[from_team]["successful"] += 1
    
    return stats

def update_candidate(current_holder, candidate, candidate_frames, prev_holder, pass_events, frame_idx):
    if current_holder is None:
        return candidate, candidate_frames, prev_holder, pass_events
    
    if candidate is None:
        return current_holder, 1, prev_holder, pass_events
    
    if current_holder["id"] == candidate["id"]:
        candidate_frames += 1
    else:
        return current_holder, 1, prev_holder, pass_events
    
    if candidate_frames == 3:
        passes_detected = detect_pass(prev_holder, candidate, frame_idx)
        prev_holder = candidate
        if passes_detected is not None:
            pass_events.append(passes_detected)
    
    return candidate, candidate_frames, prev_holder, pass_events