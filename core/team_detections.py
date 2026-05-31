import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        
    def get_clustering_model(self, image):
        image_2d = image.reshape(-1,3)
        
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1)
        kmeans.fit(image_2d)
        
        return kmeans
    
    def get_player_color(self,frame,bbox):
        x1,y1,x2,y2 = bbox
        player = frame[int(y1):int((y1+y2)//2),int(x1):int(x2)]
        
        kmeans = self.get_clustering_model(player)
        labels = kmeans.labels_
        labels = labels.reshape(player.shape[0],player.shape[1])
        
        corners = [labels[0,0],labels[0,-1],labels[-1,0],labels[-1,-1]]
        
        background_cluster = max(set(corners),key=corners.count)
        
        player_cluster = 1 - background_cluster
        
        return kmeans.cluster_centers_[player_cluster]
        
    def assign_team_color(self,frame,player_detections):
        player_color = []
        for player in player_detections:
            bbox = player["box"]
            cluster_center = self.get_player_color(frame,bbox)
            player_color.append(cluster_center)
            
        kmeans =  KMeans(n_clusters=2,init="k-means++",n_init=1)
        self.kmeans = kmeans.fit(player_color)
            
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]
                
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        color = self.get_player_color(frame,player_bbox).reshape(1,-1)
        
        which_team = self.kmeans.predict(color)
        
        which_team = which_team + 1
        
        self.player_team_dict[player_id] = which_team[0]
        
        return self.player_team_dict[player_id]
        
    
        
        