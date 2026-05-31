from speed import get_center
import numpy as np
from math import sqrt

def get_possession(ball,players,transformer):
    if len(ball) == 0:
        return None
    
    ball_x, ball_y = get_center(ball[0]["box"])
    ball_real = transformer.transform_points(np.array([[ball_x, ball_y]]))[0]
    
    min_distance = float('inf')
    closest_player = None    
    for player in players:
        player_x, player_y = get_center(player["box"])
    
        player_real = transformer.transform_points(np.array([[player_x, player_y]]))[0]
    
        distance_between_ball_player = sqrt((player_real[0]-ball_real[0])**2 + (player_real[1]-ball_real[1])**2)
        
        if distance_between_ball_player< min_distance:
            min_distance = distance_between_ball_player
            closest_player = player
    
    if min_distance < 1.5 :
        return closest_player["team_id"]
    
    return None


def update_possession(possession, team_ball_control):
    if possession is not None:
        team_ball_control.append(possession)
    elif len(team_ball_control) > 0:
        team_ball_control.append(team_ball_control[-1])
    return team_ball_control    
        