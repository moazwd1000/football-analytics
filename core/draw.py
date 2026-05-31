import cv2


def draw_player(frame, player, team_color, speed):
        x1,y1,x2,y2 = map(int, player["box"])
        track_id = player["id"]
        conf = player["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
        cv2.putText(frame, f"P{track_id} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{speed:.1f} km/h", (x1, y1-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    
def draw_ball(frame, ball):
        x1, y1, x2, y2 = map(int, ball["box"])
        conf = ball["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

    
def draw_goalkeeper(frame, gk):
        x1, y1, x2, y2 = map(int, gk["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"GK{gk['id']}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
def draw_referee(frame, ref):
        x1, y1, x2, y2 = map(int, ref["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"REF{ref['id']}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
def draw_possession(frame, possession, w):
        if possession is not None:
            cv2.putText(frame, f"Possession: Team {possession}", (w//2 - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Possession: None", (w//2 - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
def draw_pass_stats(frame, stats):
    team1_text = f"Team 1: {stats[1]['total']} passes | {stats[1]['successful']} successful"
    team2_text = f"Team 2: {stats[2]['total']} passes | {stats[2]['successful']} successful"
    
    h, w = frame.shape[:2]
    
    # background box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h-80), (500, h-10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # text
    cv2.putText(frame, team1_text, (20, h-55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, team2_text, (20, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)