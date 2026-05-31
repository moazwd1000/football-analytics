# Football Analytics

A real-time football match analysis system that tracks players, detects teams, measures speed, monitors possession, detects passes, and generates heatmaps — all from a single match video.

---

## Demo

> <img width="1920" height="1080" alt="output - frame at 0m0s" src="https://github.com/user-attachments/assets/098f741b-24d3-4ac8-bf68-baf0becd4dd1" />


---

## Features

- **Player & Ball Detection** — YOLO-based detection with ByteTrack multi-object tracking
- **Team Assignment** — Automatic team separation using jersey color clustering (KMeans)
- **Speed Estimation** — Real-world speed in km/h using homographic projection
- **Possession Tracking** — Frame-by-frame ball possession per team
- **Pass Detection** — Detects successful and unsuccessful passes per team
- **Pitch Segmentation** — SAM-based pitch mask to filter out off-pitch detections
- **Heatmaps** — Per-team positional heatmaps overlaid on a 2D pitch
- **Homography** — Maps pixel coordinates to real-world pitch coordinates via keypoint detection

---

## How It Works

```
Video Input
    ↓
YOLO Detection + ByteTrack
    ↓
Pitch Segmentation (SAM) → Filter off-pitch detections
    ↓
Homography (pitch keypoints) → Real-world coordinates
    ↓
Team Assignment (KMeans on jersey color)
    ↓
Speed / Possession / Pass Detection
    ↓
Annotated Video Output + Heatmaps
```

---

## Tech Stack

| Component | Library |
|---|---|
| Object Detection | Ultralytics YOLOv8 |
| Object Tracking | ByteTrack |
| Pitch Segmentation | Ultralytics SAM (MobileSAM) |
| Homography | OpenCV + Roboflow Inference |
| Team Clustering | scikit-learn KMeans |
| Pitch Config | `sports` (SoccerPitchConfiguration) |
| Video I/O | OpenCV |
| Dashboard | Streamlit *(coming soon)* |
| NLP | HuggingFace + LangChain + ChromaDB *(coming soon)* |
| Deployment | Docker *(coming soon)* |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/football-analytics.git
cd football-analytics
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model weights**

- Detection model: place at `models/best2.pt`
- SAM model: place at `models/mobile_sam.pt`
- Pitch keypoint model: loaded automatically via Roboflow Inference (`football-field-detection-f07vi/14`)

**4. Set your video path**

In `main.py`, update:
```python
cap = cv2.VideoCapture("path/to/your/video.mp4")
```

**5. Run**
```bash
python main.py
```

Outputs:
- `output.mp4` — annotated video
- `heatmap_team1.jpg` — Team 1 positional heatmap
- `heatmap_team2.jpg` — Team 2 positional heatmap

---

## Known Limitations

- Single fixed camera only — pan/zoom breaks the homography
- Team assignment is based on jersey color — may struggle with similar-colored kits
- Ball detection confidence is lower than player detection by design (threshold 0.1)
- Homography is recalculated every second — may drift on low-quality footage

---

## Roadmap

**Phase 1 — Core Analytics** ✅
- [x] Player & ball detection
- [x] Team assignment
- [x] Speed estimation
- [x] Possession tracking
- [x] Pass detection
- [x] Heatmaps (per team)

**Phase 2 — Dashboard** 🔄
- [ ] Streamlit frontend — 4 pages: match overview, player analysis, team analysis, match query
- [ ] Per-player speed over time charts
- [ ] Pass network visualization
- [ ] Individual player heatmaps

**Phase 3 — NLP Layer** 🔄
- [ ] Convert match events to natural language
- [ ] Sentence embeddings with HuggingFace sentence-transformers
- [ ] Vector storage with ChromaDB
- [ ] Natural language match querying

**Phase 4 — RAG + LLM** 🔄
- [ ] LangChain connecting to ChromaDB
- [ ] LLM answers questions about match data
- [ ] Connected to Streamlit query page

**Phase 5 — Deployment** 🔄
- [ ] Dockerize everything
- [ ] Deploy API to Railway or Render
- [ ] Deploy NLP demo to HuggingFace Spaces

---

## Project Structure

```
football-analytics/
├── core/
│   ├── main.py               # Entry point
│   ├── detection.py          # YOLO detection + ByteTrack
│   ├── team_detections.py    # KMeans team assignment
│   ├── homography.py         # Pitch homography + view transformer
│   ├── segmentation.py       # SAM pitch mask
│   ├── speed.py              # Speed calculation
│   ├── possession.py         # Possession tracking
│   ├── passes.py             # Pass detection
│   ├── heatmaps.py           # Heatmap generation
│   └── draw.py               # Frame annotation
├── models/                   # Model weights (not included in repo)
├── data/                     # Input videos (not included in repo)
├── .gitignore
└── README.md
```
