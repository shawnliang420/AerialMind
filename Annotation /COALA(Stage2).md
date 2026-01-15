# Interactive RMOT Annotation Editor

An intuitive and powerful tool for **Referring Multi-Object Tracking (RMOT)** annotation tasks. This application provides a user-friendly interface for selecting and tracking objects across video sequences, making it easy to create high-quality training data for RMOT models.

<div align=center><img src="../Figs/Stage2.png"/></div>

## 🌟 Features

### 🎯 Object Selection & Tracking

- **Click-to-Select**: Left-click on objects to select them for tracking
- **Right-Click Deselect**: Right-click to deselect objects
- **Persistent Tracking**: Once selected, objects remain tracked until manually deselected or task completion
- **Visual Feedback**: Selected objects are highlighted with distinct colors
- **Multi-Object Support**: Select and track multiple objects simultaneously

### 🎮 Navigation & Playback

- **Frame Navigation**: Navigate through video sequences frame by frame
- **Auto-Play**: Automatic playback with adjustable speed control (1-100ms intervals)
- **Jump to Frame**: Quick navigation to specific frames
- **Keyboard Shortcuts**: Efficient navigation using arrow keys and shortcuts

### 🔍 Display & Interaction

- **Zoom & Pan**: Ctrl+Scroll to zoom, middle-click drag to pan for detailed inspection
- **Object Information**: Display object IDs, scores, and tracking status
- **Score Filtering**: Toggle visibility of objects with score=0
- **Label Toggle**: Show/hide object ID and score labels

### 🛠️ Dual Operating Modes

#### 📝 JSON Mode (Expression Annotation)

- **Purpose**: Create expression-based annotations for RMOT tasks
- **Workflow**: Select objects and provide text descriptions
- **Output**: JSON files containing object selections and descriptions
- **Use Case**: Training data for referring expression understanding

#### 🔀 Merge Mode (Object Merging)

- **Purpose**: Merge multiple bounding boxes into unified objects
- **Auto-Tracking**: Automatic tracking of merged objects across frames
- **Visual Feedback**: Special highlighting for merged and tracking objects
- **Advanced Features**: Multi-object merging with intelligent ID management

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Required packages:

```bash
pip install opencv-python numpy tkinter argparse
```

### Quick Start

1. Clone or download the repository
2. Install dependencies
3. Run the application:

```bash
python interactive_mot_editor.py
```

## 📖 Usage Guide

### 1. Launch Application

Run the script and select your preferred mode:

- **JSON Mode**: For expression annotation tasks
- **Merge Mode**: For object merging and tracking

### 2. Load Dataset

- Click "Open Image Directory" to select your image sequence folder
- The application will automatically search for corresponding annotation files
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 3. Object Selection (JSON Mode)

1. **Select Objects**: Left-click on objects to select them
2. **Add Description**: Enter a descriptive text for the selected objects
3. **Navigate**: Use playback controls or keyboard shortcuts to move through frames
4. **Deselect**: Right-click on objects to remove them from selection
5. **Save**: Export annotations as JSON files

### 4. Object Merging (Merge Mode)

1. **Start Merge**: Double-click on the first object to begin merging
2. **Complete Merge**: Double-click on the second object to create merged box
3. **Auto-Tracking**: System automatically tracks merged objects across frames
4. **Stop Tracking**: Right-double-click to stop tracking specific or all objects
5. **Save**: Export merged annotations to text files

## ⌨️ Keyboard Shortcuts

| Key      | Action               |
| -------- | -------------------- |
| `←`      | Previous frame       |
| `→`      | Next frame           |
| `Home`   | Jump to first frame  |
| `Space`  | Play/Pause auto-play |
| `Escape` | Exit application     |

## 🖱️ Mouse Controls

### Basic Interaction

- **Left Click**: Select object
- **Right Click**: Deselect object
- **Double Left Click**: Start merge (Merge mode only)
- **Double Right Click**: Stop tracking (Merge mode only)

### Zoom & Pan

- **Ctrl + Scroll**: Zoom in/out
- **Middle Click + Drag**: Pan around zoomed image
- **Reset Zoom Button**: Return to original view

## 📁 File Structure

### Input Files

```
dataset/
├── sequences/
│   └── video_name/
│       ├── img000001.jpg
│       ├── img000002.jpg
│       └── ...
└── annotations/
    └── video_name.txt
```

### Output Files

#### JSON Mode Output

```json
{
  "label": {
    "1": [object_id1, object_id2],
    "2": [object_id1],
    "...": "..."
  },
  "ignore": {},
  "video_name": "sequence_name",
  "sentence": "description text"
}
```

#### Merge Mode Output

MOT format text file with merged object annotations:

```
frame_id,object_id,x,y,w,h,score,category,truncation,occlusion
```



## 🐛 Troubleshooting

### Common Issues

1. **Image not loading**: Check file format and path
2. **Annotation file not found**: Ensure proper file structure
3. **Performance issues**: Reduce image resolution or close other applications
4. **Selection not working**: Check if object has score > 0 (or enable score=0 display)

### System Requirements

- **Memory**: 4GB+ RAM recommended for large sequences
- **Display**: 1600x900+ resolution for optimal interface experience
- **Storage**: Sufficient space for image sequences and output files

## Citing AerialMind
If you find AerialMind useful in your research, please consider citing:

```bibtex
@article{chen2025aerialmind,
  title={AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios},
  author={Chen, Chenglizhao and Liang, Shaofeng and Guan, Runwei and Sun, Xiaolou and Zhao, Haocheng and Jiang, Haiyun and Huang, Tao and Ding, Henghui and Han, Qing-Long},
  journal={arXiv preprint arXiv:2511.21053},
  year={2025}
}