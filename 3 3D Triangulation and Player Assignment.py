import cv2
import numpy as np
import mediapipe as mp
import glob

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Step 1: Camera Calibration (Using precomputed values)
CHESSBOARD_SIZE = (7, 6)
SQUARE_SIZE = 0.025

def find_chessboard_corners(images):
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    objpoints, imgpoints = [], []
    
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    return objpoints, imgpoints, gray.shape[::-1]

# Load calibration images
left_images = glob.glob('left_cam/*.jpg')
right_images = glob.glob('right_cam/*.jpg')

objpoints, imgpointsL, img_sizeL = find_chessboard_corners(left_images)
_, imgpointsR, img_sizeR = find_chessboard_corners(right_images)

# Calibrate both cameras
retL, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpointsL, img_sizeL, None, None)
retR, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpointsR, img_sizeR, None, None)

# Stereo Calibration (Find R & T)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_FIX_INTRINSIC
ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, K1, dist1, K2, dist2, img_sizeL,
    criteria=criteria, flags=flags
)

# Compute Projection Matrices
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, dist1, K2, dist2, img_sizeL, R, T)

print("Calibration Complete.")

# Step 2: Open synchronized camera streams
cap_left = cv2.VideoCapture(0)  # Left camera
cap_right = cv2.VideoCapture(1)  # Right camera

# Synchronization function
def sync_frames(cap_left, cap_right):
    while True:
        retL, frameL = cap_left.read()
        retR, frameR = cap_right.read()
        
        if not retL or not retR:
            continue  # Skip if frames are not available
        
        timestampL = cap_left.get(cv2.CAP_PROP_POS_MSEC)
        timestampR = cap_right.get(cv2.CAP_PROP_POS_MSEC)
        
        # Adjust frame timing if timestamps differ
        if abs(timestampL - timestampR) < 5:  # Tolerance of 5ms
            return frameL, frameR

# Step 3: Detect 2D keypoints with MediaPipe
def get_pose_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.append((lm.x * frame.shape[1], lm.y * frame.shape[0]))  # Convert to pixel coordinates
    
    return keypoints

# Step 4: Triangulate 2D Keypoints into 3D Space
def triangulate_points(kpL, kpR, P1, P2):
    """ Triangulate 2D keypoints from both views into 3D points. """
    kpL = np.array(kpL, dtype=np.float32).T  # (2, N)
    kpR = np.array(kpR, dtype=np.float32).T  # (2, N)

    homogeneous_3D = cv2.triangulatePoints(P1, P2, kpL, kpR)
    points_3D = cv2.convertPointsFromHomogeneous(homogeneous_3D.T)  # Convert to 3D coordinates

    return points_3D.squeeze()  # Remove unnecessary dimensions

# Step 5: Assign Keypoints to Players (Batsman & Bowler)
def assign_players(points_3D):
    """ Assigns keypoints to the batsman or bowler based on X-axis position. """
    if len(points_3D) == 0:
        return None, None

    x_coords = points_3D[:, 0]  # Extract X positions
    midpoint = np.mean(x_coords)  # Find center X position

    batsman = points_3D[x_coords < midpoint]  # Closer player
    bowler = points_3D[x_coords >= midpoint]  # Farther player

    return batsman, bowler

while True:
    # Synchronize camera frames
    frame_left, frame_right = sync_frames(cap_left, cap_right)
    
    # Extract 2D keypoints
    keypointsL = get_pose_keypoints(frame_left)
    keypointsR = get_pose_keypoints(frame_right)

    if keypointsL and keypointsR:
        # Ensure equal number of keypoints for triangulation
        min_points = min(len(keypointsL), len(keypointsR))
        keypointsL, keypointsR = keypointsL[:min_points], keypointsR[:min_points]

        # Convert 2D keypoints to 3D
        points_3D = triangulate_points(keypointsL, keypointsR, P1, P2)

        # Assign keypoints to players
        batsman, bowler = assign_players(points_3D)

        # Display 3D keypoints
        print("\n3D Keypoints:")
        print("Batsman:", batsman if batsman is not None else "Not detected")
        print("Bowler:", bowler if bowler is not None else "Not detected")

    # Draw keypoints on frames
    for (x, y) in keypointsL:
        cv2.circle(frame_left, (int(x), int(y)), 5, (0, 255, 0), -1)

    for (x, y) in keypointsR:
        cv2.circle(frame_right, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Display synchronized frames
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
