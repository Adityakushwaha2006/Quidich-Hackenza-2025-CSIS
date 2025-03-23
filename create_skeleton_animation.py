import cv2
import mediapipe as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Local paths 
c233_path = r'233_im'
c235_path = r'235_im'
intrinsic_path = r'intrinsic.json'
extrinsic_path = r'extrinsic.json'

# Create output directory
os.makedirs('output', exist_ok=True)

print("Loading camera calibration data...")
# Load camera calibration data
with open(intrinsic_path, 'r') as f:
    intrinsic_data = json.load(f)

with open(extrinsic_path, 'r') as f:
    extrinsic_data = json.load(f)

# Extract camera matrices
K1 = np.array(intrinsic_data['C233']['camera_matrix'])
D1 = np.array(intrinsic_data['C233']['distortion_coefficients'])
K2 = np.array(intrinsic_data['C235']['camera_matrix'])
D2 = np.array(intrinsic_data['C235']['distortion_coefficients'])

# Extract rotation and translation matrices
R1 = cv2.Rodrigues(np.array(extrinsic_data['rotation_vectors']['C233']))[0]
T1 = np.array(extrinsic_data['translation_vectors']['C233']).reshape(3, 1)
R2 = cv2.Rodrigues(np.array(extrinsic_data['rotation_vectors']['C235']))[0]
T2 = np.array(extrinsic_data['translation_vectors']['C235']).reshape(3, 1)

# Compute projection matrices
P1 = K1 @ np.hstack((R1, T1))
P2 = K2 @ np.hstack((R2, T2))

print("Setting up MediaPipe pose detector...")
# MediaPipe pose detector with minimal settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,  # Use lowest complexity for stability
    enable_segmentation=False,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Skeleton connections for 3D visualization
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),  # Nose to eyes and eyes to ears
    
    # Torso
    (11, 12), (12, 24), (24, 23), (23, 11),  # Hips and shoulders
    
    # Arms
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),  # Left arm
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16),  # Right arm
    
    # Legs
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),  # Right leg
    
    # Shoulders to hips
    (11, 23), (12, 24),
    
    # Neck and face
    (11, 0), (12, 0)
]

def triangulate_point(P1, P2, point1, point2):
    """Triangulate a 3D point from two 2D points and projection matrices"""
    A = np.array([
        point1[0] * P1[2] - P1[0],
        point1[1] * P1[2] - P1[1],
        point2[0] * P2[2] - P2[0],
        point2[1] * P2[2] - P2[1]
    ])
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    return X[:3] / X[3]

def process_frame(frame_number):
    """Process a single frame for pose estimation and triangulation"""
    # Build image filenames
    file1 = f"HPUP_033_1_1_1_L_CAM-05_{frame_number:07d}.jpeg"
    file2 = f"HPUP_033_1_1_1_L_CAM-02_{frame_number:07d}.jpeg"
    
    # Read frames
    frame1 = cv2.imread(os.path.join(c233_path, file1))
    frame2 = cv2.imread(os.path.join(c235_path, file2))
    
    if frame1 is None or frame2 is None:
        print(f"Frame {frame_number}: Files not found")
        return None
        
    print(f"Processing frame {frame_number}")
    
    # Make copies for visualization
    vis_frame1 = frame1.copy()
    vis_frame2 = frame2.copy()
    
    # Process with MediaPipe
    try:
        results_pose1 = pose.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        results_pose2 = pose.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        
        landmarks_detected1 = results_pose1.pose_landmarks is not None
        landmarks_detected2 = results_pose2.pose_landmarks is not None
        
        print(f"Frame {frame_number}: Landmarks detected - Camera 1: {landmarks_detected1}, Camera 2: {landmarks_detected2}")
        
        # Draw landmarks on visualization frames
        if landmarks_detected1:
            mp_drawing.draw_landmarks(
                vis_frame1, 
                results_pose1.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
        if landmarks_detected2:
            mp_drawing.draw_landmarks(
                vis_frame2, 
                results_pose2.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
        
        # Save visualization frames
        cv2.imwrite(f"output/cam1_frame{frame_number}.jpg", vis_frame1)
        cv2.imwrite(f"output/cam2_frame{frame_number}.jpg", vis_frame2)
        
        # Triangulate 3D points if landmarks detected in both cameras
        if landmarks_detected1 and landmarks_detected2:
            points_3d = []
            for i in range(33):  # MediaPipe detects 33 landmarks
                try:
                    # Get 2D points from each camera view
                    point1 = np.array([
                        results_pose1.pose_landmarks.landmark[i].x * frame1.shape[1],
                        results_pose1.pose_landmarks.landmark[i].y * frame1.shape[0]
                    ])
                    point2 = np.array([
                        results_pose2.pose_landmarks.landmark[i].x * frame2.shape[1],
                        results_pose2.pose_landmarks.landmark[i].y * frame2.shape[0]
                    ])
                    
                    # Triangulate 3D point
                    point_3d = triangulate_point(P1, P2, point1, point2)
                    points_3d.append(point_3d)
                except Exception as e:
                    print(f"Frame {frame_number}: Error triangulating point {i}")
                    # Insert a zero point for missing landmarks
                    points_3d.append(np.zeros(3))
            
            points_3d = np.array(points_3d)
            print(f"Frame {frame_number}: Generated 3D points with shape {points_3d.shape}")
            return {'frame': frame_number, 'points': points_3d}
        else:
            print(f"Frame {frame_number}: Cannot triangulate - landmarks not detected in both cameras")
            return None
    except Exception as e:
        print(f"Frame {frame_number}: Error in processing: {e}")
        return None

def draw_skeleton_3d(ax, points, color='blue', alpha=0.7, linewidth=2):
    """Draw a 3D skeleton connecting landmarks according to SKELETON_CONNECTIONS"""
    if points.ndim != 2 or points.shape[1] != 3:
        print(f"Invalid points array shape: {points.shape}")
        return
    
    # Draw points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=20, alpha=alpha)
    
    # Draw connections
    for connection in SKELETON_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 < len(points) and idx2 < len(points):
            # Get points coordinates
            x1, y1, z1 = points[idx1]
            x2, y2, z2 = points[idx2]
            
            # Skip if either point is zero (missing landmark)
            if np.all(np.isclose([x1, y1, z1], 0)) or np.all(np.isclose([x2, y2, z2], 0)):
                continue
            
            # Draw line
            ax.plot([x1, x2], [y1, y2], [z1, z2], c=color, linewidth=linewidth, alpha=alpha)

def create_animation(all_3d_points):
    """Create a 3D animation from the collected 3D points"""
    if not all_3d_points:
        print("No 3D points to animate")
        return
    
    print(f"Creating animation with {len(all_3d_points)} frames...")
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find the bounds of all points to set consistent axis limits
    all_points = np.vstack([data['points'] for data in all_3d_points])
    all_points = all_points[~np.all(all_points == 0, axis=1)]  # Remove zero points
    
    if len(all_points) == 0:
        print("No valid points found for animation")
        return
    
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)
    
    # Add some padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    def update(frame):
        ax.clear()
        frame_data = all_3d_points[frame]
        
        draw_skeleton_3d(ax, frame_data['points'], color='blue', alpha=0.8, linewidth=2)
        
        # Set title with frame number
        ax.set_title(f'3D Cricket Player Pose - Frame {frame_data["frame"]}')
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set consistent axis limits
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
        
        # Set consistent viewpoint
        ax.view_init(elev=20, azim=45)
        
        return ax
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(all_3d_points), interval=200, blit=False)
    
    # Save animation
    anim.save('output/3d_cricket_skeleton.gif', writer='pillow', fps=5, dpi=80)
    print("Animation saved as 'output/3d_cricket_skeleton.gif'")
    
    # Close figure
    plt.close(fig)

def main():
    # Process frames that have been verified to work
    frame_range = list(range(290, 311))
    print(f"Processing frames {frame_range[0]} to {frame_range[-1]}...")
    
    # Collect 3D points
    all_3d_points = []
    for frame_number in frame_range:
        result = process_frame(frame_number)
        if result is not None:
            all_3d_points.append(result)
    
    print(f"Successfully processed {len(all_3d_points)} frames with valid 3D points")
    
    if all_3d_points:
        # Create a static visualization for verification
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use a different color for each frame
        colors = plt.cm.jet(np.linspace(0, 1, len(all_3d_points)))
        
        for i, data in enumerate(all_3d_points):
            draw_skeleton_3d(ax, data['points'], color=colors[i], alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Cricket Player Pose - All Frames')
        ax.view_init(elev=20, azim=45)
        
        plt.savefig('output/3d_cricket_static.png')
        print("Static visualization saved as 'output/3d_cricket_static.png'")
        plt.close(fig)
        
        # Create animation
        create_animation(all_3d_points)
        
        # Save 3D points to a JSON file for later use
        print("Saving 3D points to JSON file...")
        json_data = []
        for data in all_3d_points:
            json_data.append({
                'frame': int(data['frame']),
                'points': data['points'].tolist()
            })
        
        with open('output/3d_points.json', 'w') as f:
            json.dump(json_data, f)
        print("3D points saved to 'output/3d_points.json'")
    else:
        print("No frames with valid 3D points were found")
    
    # Clean up
    pose.close()
    print("Processing complete")

if __name__ == "__main__":
    main() 