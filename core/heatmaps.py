import numpy as np
import cv2
from sports.configs.soccer import SoccerPitchConfiguration
from speed import get_center
config = SoccerPitchConfiguration() 


def generate_heatmap(positions, pitch_width=105, pitch_height=68, 
                     output_size=(680, 1050)):
    
    heatmap = np.zeros((680, 1050),dtype=np.float32)
    
    for x,y in positions:
        px = int((x/pitch_width) * output_size[1])
        py = int((y/pitch_height) * output_size[0])
        
        if  0 < px < output_size[1] and  0 < py < output_size[0]:
            heatmap[py,px] +=1
    
    heatmap = cv2.GaussianBlur(heatmap,(31,31),0)
    
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    
        
    colored_image = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    pitch = draw_pitch(output_size)
    blended = cv2.addWeighted(pitch, 0.5, colored_image, 0.5, 0)
    return blended
        
        
def draw_pitch(output_size=(680, 1050)):
    image = np.ones((680,1050,3),dtype = np.uint8) * 34
    pixels = []
    for x,y in config.vertices:         
        px = int((x/12000)*1050)
        py = int((y/7000)*680)
        pixels.append((px,py))
        
        
    for i, j in config.edges:
        cv2.line(image, pixels[i-1], pixels[j-1], (255, 255, 255), 2)
    
    return image

def update_positions(player, transformer, all_positions,player_id ):
    x_c, y_b = get_center(player["box"])
    real_pos = transformer.transform_points(np.array([[x_c, y_b]]))[0]
    all_positions[player_id].append((real_pos[0], real_pos[1]))
    return all_positions


def generate_team_heatmaps(all_positions, team_assignments, pitch_width=105, pitch_height=68, output_size=(680, 1050)):
    team1_positions = []
    team2_positions = []
    
    for player_id, positions in all_positions.items():
        team = team_assignments.get(player_id)
        if team == 1:
            team1_positions.extend(positions)
        elif team == 2:
            team2_positions.extend(positions)
    
    heatmap_1 = generate_heatmap(team1_positions, pitch_width, pitch_height, output_size)
    heatmap_2 = generate_heatmap(team2_positions, pitch_width, pitch_height, output_size)
    
    return heatmap_1, heatmap_2