import cv2
import numpy as np
import mediapipe as mp
import glob
import threading
import time
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MediaPipe Pose setup (enable GPU acceleration)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Enable CUDA if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# Step 1: Camera Calibration (Precomputed values used)
# Load stereo calibration parameters (assumed precomputed)
K1 = np.load("K1.npy")
K2 = np.load("K2.npy")
dist1 = np.load("dist1.npy")
dist2 = np.load("dist2.npy")
R = np.load("R.npy")
T = np.load("T.npy")

# Compute Projection Matrices
_, _, P1, P2, _, _, _ = cv2.stereoRectify(K1, dist1, K2, dist2, (1280, 720), R, T)

print("Calibration Loaded.")

# Step 2: Multi-threaded Camera Capture
cap_left = cv2.VideoCapture(0)  # Left camera
cap_right = cv2.VideoCapture(1)  # Right camera

frameL, frameR = None, None
lock = threading.Lock()

def capture_frames():
    global frameL, frameR
    while True:
        retL, tempL = cap_left.read()
        retR, tempR = cap_right.read()

        if retL and retR:
            with lock:
                frameL, frameR = tempL, tempR

threading.Thread(target=capture_frames, daemon=True).start()

# Step 3: Pose Estimation
def get_pose_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x * frame.shape[1], lm.y * frame.shape[0]])  # Convert to pixel coordinates
    
    return np.array(keypoints, dtype=np.float32)

# Step 4: Triangulation (Optimized with NumPy)
def triangulate_points(kpL, kpR, P1, P2):
    if len(kpL) == 0 or len(kpR) == 0:
        return np.array([])

    kpL = np.array(kpL, dtype=np.float32).T  # (2, N)
    kpR = np.array(kpR, dtype=np.float32).T  # (2, N)

    homogeneous_3D = cv2.triangulatePoints(P1, P2, kpL, kpR)
    points_3D = cv2.convertPointsFromHomogeneous(homogeneous_3D.T).squeeze()  # Convert to 3D
    
    return points_3D

# Step 5: Assign Players (Batsman & Bowler)
def assign_players(points_3D):
    if points_3D.size == 0:
        return None, None

    x_coords = points_3D[:, 0]  # Extract X positions
    midpoint = np.mean(x_coords)  # Find center X position

    batsman = points_3D[x_coords < midpoint]  # Closer player
    bowler = points_3D[x_coords >= midpoint]  # Farther player

    return batsman, bowler

# Step 6: 3D Visualization
def plot_3D_keypoints(batsman, bowler):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if batsman is not None:
        ax.scatter(batsman[:, 0], batsman[:, 1], batsman[:, 2], c='blue', label="Batsman")
    if bowler is not None:
        ax.scatter(bowler[:, 0], bowler[:, 1], bowler[:, 2], c='red', label="Bowler")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# Main Loop
while True:
    with lock:
        if frameL is None or frameR is None:
            continue
        frame_left, frame_right = frameL.copy(), frameR.copy()

    # Get synchronized keypoints
    keypointsL = get_pose_keypoints(frame_left)
    keypointsR = get_pose_keypoints(frame_right)

    if keypointsL.size > 0 and keypointsR.size > 0:
        # Ensure equal keypoints for triangulation
        min_points = min(len(keypointsL), len(keypointsR))
        keypointsL, keypointsR = keypointsL[:min_points], keypointsR[:min_points]

        # Convert 2D keypoints to 3D
        points_3D = triangulate_points(keypointsL, keypointsR, P1, P2)

        # Assign keypoints to players
        batsman, bowler = assign_players(points_3D)

        # Display 3D keypoints in Matplotlib
        plot_3D_keypoints(batsman, bowler)

    # Draw 2D keypoints
    for (x, y) in keypointsL:
        cv2.circle(frame_left, (int(x), int(y)), 5, (0, 255, 0), -1)

    for (x, y) in keypointsR:
        cv2.circle(frame_right, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Show synchronized frames
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
