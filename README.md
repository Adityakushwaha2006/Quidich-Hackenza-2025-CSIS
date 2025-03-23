# Cricket Player 3D Pose Estimation

This project uses computer vision and multi-view geometry to track cricket players from different camera angles and reconstruct their 3D poses.

## Setup Instructions

### 1. Create a Python 3.10 virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

On Windows:
```bash
.\venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the script

```bash
python Iter5_bluedot_ani.py
```

## Project Structure

- `233_im/` - Images from camera angle 1
- `235_im/` - Images from camera angle 2
- `intrinsic.json` - Camera intrinsic parameters
- `extrinsic.json` - Camera extrinsic parameters
- `Iter5_bluedot_ani.py` - Main script for 3D pose estimation
- `requirements.txt` - Required Python packages

## Output

The script will:
1. Process frames from both camera angles
2. Detect player poses using MediaPipe
3. Triangulate 3D positions
4. Generate a static 3D visualization (`3d_pose_static.png`)
5. Create an animated visualization (`3d_pose_animation.gif`)

## Note

This script was adapted from a Google Colab notebook to run locally in VS Code. 