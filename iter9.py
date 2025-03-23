import cv2
import mediapipe as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib import cm
from scipy.spatial import ConvexHull
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import art3d

# Paths
c233_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\233_im'
c235_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\235_im'
intrinsic_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\intrinsic.json'
extrinsic_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\extrinsic.json'
output_dir = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\processed_frames'
os.makedirs(output_dir, exist_ok=True)

# Load intrinsic and extrinsic data
with open(intrinsic_path, 'r') as f:
    intrinsic_data = json.load(f)

with open(extrinsic_path, 'r') as f:
    extrinsic_data = json.load(f)

# Extract camera matrices and distortion coefficients
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

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

# Define custom connections for different body parts
# HEAD connections
HEAD_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
    (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
    (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
    (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
    (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
    (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
    (mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT),
    (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER),
])

# TORSO connections
TORSO_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
])

# ARM connections
ARM_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_THUMB),
    (mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.LEFT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_THUMB),
    (mp_pose.PoseLandmark.RIGHT_PINKY, mp_pose.PoseLandmark.RIGHT_INDEX),
])

# LEG connections
LEG_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
    (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
    (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
])

# Define custom drawing styles for different body parts
head_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)    # Yellow for head
torso_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)     # Blue for torso
arm_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)       # Green for arms
leg_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)       # Red for legs

# Custom landmark styles for different body parts
head_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
torso_landmark_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3)
arm_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
leg_landmark_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)

# Define body part color schemes for 3D visualization
BODY_COLORS = {
    "head": {
        "color": "gold",
        "alpha": 0.9,
        "edge_color": "darkgoldenrod",
        "marker": "o",
        "s": 45  # Size of markers
    },
    "torso": {
        "color": "royalblue",
        "alpha": 0.85,
        "edge_color": "navy",
        "marker": "s",  # Square markers for torso
        "s": 60
    },
    "arm": {
        "color": "forestgreen",
        "alpha": 0.8,
        "edge_color": "darkgreen",
        "marker": "o",
        "s": 40
    },
    "leg": {
        "color": "firebrick",
        "alpha": 0.85,
        "edge_color": "darkred",
        "marker": "o",
        "s": 50
    }
}

# Define connections for 3D visualization
# These are indices in the MediaPipe pose landmarks
head_connections_3d = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye to ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye to ear
    (9, 10),  # Mouth
    (7, 11), (8, 12)  # Ears to shoulders
]

torso_connections_3d = [
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 24)  # Hips
]

arm_connections_3d = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)   # Right arm
]

leg_connections_3d = [
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)   # Right leg
]

# Body part groupings for creating meshes
body_parts_grouping = {
    "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "torso": [11, 12, 23, 24],
    "left_arm": [11, 13, 15, 17, 19, 21],
    "right_arm": [12, 14, 16, 18, 20, 22],
    "left_leg": [23, 25, 27, 29, 31],
    "right_leg": [24, 26, 28, 30, 32]
}

def draw_styled_landmarks(image, results):
    if not results.pose_landmarks:
        return
    
    # Draw head
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        HEAD_CONNECTIONS,
        head_landmark_style,
        head_style
    )
    
    # Draw torso
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        TORSO_CONNECTIONS,
        torso_landmark_style,
        torso_style
    )
    
    # Draw arms
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        ARM_CONNECTIONS,
        arm_landmark_style,
        arm_style
    )
    
    # Draw legs
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        LEG_CONNECTIONS,
        leg_landmark_style,
        leg_style
    )

def triangulate_point(P1, P2, point1, point2):
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
    # Construct file names
    file1 = f"HPUP_033_1_1_1_L_CAM-05_{frame_number:07d}.jpeg"
    file2 = f"HPUP_033_1_1_1_L_CAM-02_{frame_number:07d}.jpeg"

    # Read frames from both cameras
    frame1 = cv2.imread(os.path.join(c233_path, file1))
    frame2 = cv2.imread(os.path.join(c235_path, file2))

    if frame1 is None or frame2 is None:
        print(f"Frame {frame_number} missing in one of the cameras.")
        return None

    # Process frames with MediaPipe
    results1 = pose.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    results2 = pose.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

    # Draw styled landmarks on both frames
    if results1.pose_landmarks:
        draw_styled_landmarks(frame1, results1)
    if results2.pose_landmarks:
        draw_styled_landmarks(frame2, results2)

    # Optionally save the processed frames
    cv2.imwrite(os.path.join(output_dir, f"cam233_frame_{frame_number}.jpg"), frame1)
    cv2.imwrite(os.path.join(output_dir, f"cam235_frame_{frame_number}.jpg"), frame2)

    # Initialize 3D points list
    points_3d = []
    landmarks_dict = {}
    visibility_dict = {}

    if results1.pose_landmarks and results2.pose_landmarks:
        for i in range(33):  # MediaPipe detects 33 landmarks
            # Get 2D points from both camera views
            point1 = np.array([results1.pose_landmarks.landmark[i].x * frame1.shape[1],
                               results1.pose_landmarks.landmark[i].y * frame1.shape[0]])
            point2 = np.array([results2.pose_landmarks.landmark[i].x * frame2.shape[1],
                               results2.pose_landmarks.landmark[i].y * frame2.shape[0]])

            # Get visibility scores (confidence)
            vis1 = results1.pose_landmarks.landmark[i].visibility
            vis2 = results2.pose_landmarks.landmark[i].visibility
            avg_visibility = (vis1 + vis2) / 2
            visibility_dict[i] = avg_visibility

            # Triangulate 3D point
            point_3d = triangulate_point(P1, P2, point1, point2)
            points_3d.append(point_3d)
            
            # Store landmark type for coloring in 3D visualization
            if i in body_parts_grouping["head"]:
                landmarks_dict[i] = "head"
            elif i in body_parts_grouping["torso"]:
                landmarks_dict[i] = "torso"
            elif i in body_parts_grouping["left_arm"] or i in body_parts_grouping["right_arm"]:
                landmarks_dict[i] = "arm"
            else:  # Leg landmarks
                landmarks_dict[i] = "leg"

    return np.array(points_3d), landmarks_dict, visibility_dict, frame1, frame2

def get_body_part_mesh(points, indices, hull=True):
    """Create a 3D mesh for a body part using ConvexHull if enough points are available"""
    if len(indices) < 4:  # Need at least 4 points for a 3D convex hull
        return None
    
    # Extract points for this body part
    part_points = np.array([points[i] for i in indices if i < len(points)])
    
    if len(part_points) < 4:
        return None
    
    if hull:
        try:
            # Create convex hull
            hull = ConvexHull(part_points)
            return part_points, hull
        except:
            return None
    else:
        return part_points, None

def draw_realistic_human_3d(ax, points, landmarks_dict, visibility_dict, frame_idx=0):
    """Draw a more realistic 3D human figure with meshes, cylinders, and proper lighting"""
    if len(points) == 0:
        return
    
    # Set up light source for realistic shading
    ls = LightSource(azdeg=225.0, altdeg=45.0)
    
    # 1. Draw connections with stylized cylinders
    connection_groups = [
        (head_connections_3d, BODY_COLORS["head"]),
        (torso_connections_3d, BODY_COLORS["torso"]),
        (arm_connections_3d, BODY_COLORS["arm"]),
        (leg_connections_3d, BODY_COLORS["leg"])
    ]
    
    # Draw all connections with proper widths based on body part and visibility
    for connections, style in connection_groups:
        for connection in connections:
            if connection[0] < len(points) and connection[1] < len(points):
                p1 = points[connection[0]]
                p2 = points[connection[1]]
                
                # Skip if points are too far (likely errors in pose estimation)
                if np.linalg.norm(p1 - p2) > 5:
                    continue
                
                # Get average visibility for this connection
                vis1 = visibility_dict.get(connection[0], 0.5)
                vis2 = visibility_dict.get(connection[1], 0.5)
                avg_vis = (vis1 + vis2) / 2
                
                # Adjust line width and alpha based on visibility
                lw = 3 * avg_vis + 1  # Minimum line width of 1
                alpha = min(0.9, max(0.4, avg_vis))  # Alpha between 0.4 and 0.9
                
                # Draw enhanced connection
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                       color=style["edge_color"], linewidth=lw, alpha=alpha,
                       solid_capstyle='round')
    
    # 2. Draw landmarks with 3D markers
    for i, point in enumerate(points):
        if i in landmarks_dict:
            body_part = landmarks_dict[i]
            style = BODY_COLORS[body_part]
            
            # Adjust size based on visibility
            vis = visibility_dict.get(i, 0.5)
            size = style["s"] * vis
            
            ax.scatter(point[0], point[1], point[2], 
                      c=style["color"], s=size, alpha=style["alpha"] * vis,
                      marker=style["marker"], edgecolors=style["edge_color"], 
                      linewidths=1)
    
    # 3. Try to create simplified meshes for key body parts
    try:
        # Torso mesh (always try to show)
        torso_indices = body_parts_grouping["torso"] + [13, 14]  # Include shoulder landmarks
        torso_mesh = get_body_part_mesh(points, torso_indices)
        if torso_mesh:
            part_points, hull = torso_mesh
            for simplex in hull.simplices:
                # Get triangle vertices
                triangle = part_points[simplex]
                # Create shaded surface with proper lighting
                x = triangle[:, 0]
                y = triangle[:, 1]
                z = triangle[:, 2]
                
                # Collect coordinates for a single triangular polygon
                verts = [list(zip(x, y, z))]
                
                # Shade the triangle using the specified color for torso
                ax.add_collection3d(art3d.Poly3DCollection(
                    verts, 
                    alpha=0.7,
                    color=BODY_COLORS["torso"]["color"],
                    edgecolor=BODY_COLORS["torso"]["edge_color"],
                    linewidth=0.5
                ))
    except Exception as e:
        print(f"Error creating meshes: {e}")
        # Continue without meshes
    
    # Set view to see the person from a good angle
    ax.view_init(elev=10, azim=-70)

def process_all_frames(start_frame=350, end_frame=507, step=1):
    """Process all frames and return the collected data"""
    all_3d_points = []
    all_landmarks_dicts = []
    all_visibility_dicts = []
    all_frames_cam1 = []
    all_frames_cam2 = []

    for frame_number in range(start_frame, end_frame, step):
        if frame_number % 10 == 0:  # Print every 10 frames
            print(f"Processing frame {frame_number}")
        result = process_frame(frame_number)
        
        if result is not None:
            points_3d, landmarks_dict, visibility_dict, frame1, frame2 = result
            all_3d_points.append(points_3d)
            all_landmarks_dicts.append(landmarks_dict)
            all_visibility_dicts.append(visibility_dict)
            all_frames_cam1.append(frame1)
            all_frames_cam2.append(frame2)

    return all_3d_points, all_landmarks_dicts, all_visibility_dicts, all_frames_cam1, all_frames_cam2

def create_side_by_side_visualization(frame_number, fig_size=(18, 10)):
    """Create a side-by-side visualization with camera views and 3D model"""
    fig = plt.figure(figsize=fig_size)
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3)
    
    # Camera 1 view
    ax1 = fig.add_subplot(gs[0, 0])
    if frame_number < len(all_frames_cam1):
        ax1.imshow(cv2.cvtColor(all_frames_cam1[frame_number], cv2.COLOR_BGR2RGB))
    ax1.set_title('Camera 233 View')
    ax1.axis('off')
    
    # Camera 2 view
    ax2 = fig.add_subplot(gs[0, 1])
    if frame_number < len(all_frames_cam2):
        ax2.imshow(cv2.cvtColor(all_frames_cam2[frame_number], cv2.COLOR_BGR2RGB))
    ax2.set_title('Camera 235 View')
    ax2.axis('off')
    
    # 3D view (takes more space)
    ax3 = fig.add_subplot(gs[:, 2], projection='3d')
    if frame_number < len(all_3d_points):
        points = all_3d_points[frame_number]
        landmarks_dict = all_landmarks_dicts[frame_number]
        visibility_dict = all_visibility_dicts[frame_number]
        draw_realistic_human_3d(ax3, points, landmarks_dict, visibility_dict, frame_number)
    
    # Set limits based on your data
    ax3.set_xlim([996, 1002])
    ax3.set_ylim([1004.6, 1006])
    ax3.set_zlim([98, 101])
    
    ax3.set_title(f'3D Reconstruction - Frame {frame_number+350}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Add legend
    head_patch = mpatches.Patch(color=BODY_COLORS["head"]["color"], label='Head')
    torso_patch = mpatches.Patch(color=BODY_COLORS["torso"]["color"], label='Torso')
    arm_patch = mpatches.Patch(color=BODY_COLORS["arm"]["color"], label='Arms')
    leg_patch = mpatches.Patch(color=BODY_COLORS["leg"]["color"], label='Legs')
    ax3.legend(handles=[head_patch, torso_patch, arm_patch, leg_patch], loc='upper right')
    
    # Trajectory view (shows historical motion paths)
    ax4 = fig.add_subplot(gs[1, :2], projection='3d')
    draw_trajectory(ax4, frame_number)
    ax4.set_title('Motion Trajectory')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    plt.tight_layout()
    return fig

def draw_trajectory(ax, current_frame, trail_length=10):
    """Draw a trajectory of important body parts (head, hands, feet)"""
    # Key points to track
    key_points = {
        "Head": 0,  # Nose
        "Left Hand": 15,  # Left wrist
        "Right Hand": 16,  # Right wrist
        "Left Foot": 31,  # Left foot index
        "Right Foot": 32   # Right foot index
    }
    
    # Colors for each trajectory
    colors = {
        "Head": "gold",
        "Left Hand": "lime",
        "Right Hand": "cyan",
        "Left Foot": "magenta",
        "Right Foot": "orange"
    }
    
    # Define the range of frames to include in the trajectory
    start_frame = max(0, current_frame - trail_length)
    end_frame = current_frame + 1  # Include the current frame
    
    # Plot trajectories for each key point
    for part_name, point_idx in key_points.items():
        xs, ys, zs = [], [], []
        
        for i in range(start_frame, end_frame):
            if i < len(all_3d_points) and point_idx < len(all_3d_points[i]):
                point = all_3d_points[i][point_idx]
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
        
        if xs:
            # Plot trajectory with increasing thickness and opacity
            for i in range(len(xs)-1):
                # Calculate alpha and line width based on position in sequence
                progress = i / max(1, len(xs)-1)
                alpha = 0.3 + 0.7 * progress
                lw = 1 + 3 * progress
                
                ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2], 
                       color=colors[part_name], alpha=alpha, linewidth=lw)
            
            # Plot the current position with a larger marker
            ax.scatter(xs[-1], ys[-1], zs[-1], 
                      c=colors[part_name], s=50, label=part_name)
    
    # Set appropriate limits
    ax.set_xlim([996, 1002])
    ax.set_ylim([1004.6, 1006])
    ax.set_zlim([98, 101])
    
    # Add legend
    ax.legend(loc='upper right')

    # Set a more dynamic viewing angle
    ax.view_init(elev=20, azim=45)

# Main execution

print("Processing all frames...")
all_3d_points, all_landmarks_dicts, all_visibility_dicts, all_frames_cam1, all_frames_cam2 = process_all_frames(
    start_frame=350, 
    end_frame=507, 
    step=2  # Process every other frame for speed
)
print(f"Processed {len(all_3d_points)} frames.")

# Create a single frame visualization for testing
print("Creating visualization...")
test_frame = 0
fig = create_side_by_side_visualization(test_frame)
plt.savefig(os.path.join(output_dir, 'side_by_side_test.png'), dpi=300)
plt.close(fig)

# Create animation
print("Creating animation...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    if frame < len(all_3d_points):
        points = all_3d_points[frame]
        landmarks_dict = all_landmarks_dicts[frame]
        visibility_dict = all_visibility_dicts[frame]
        draw_realistic_human_3d(ax, points, landmarks_dict, visibility_dict, frame)
    
    # Set fixed limits to prevent camera movement
    ax.set_xlim([996, 1002])
    ax.set_ylim([1004.6, 1006])
    ax.set_zlim([98, 101])
    
    ax.set_title(f'3D Pose Estimation - Frame {frame*2+350}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add legend
    head_patch = mpatches.Patch(color=BODY_COLORS["head"]["color"], label='Head')
    torso_patch = mpatches.Patch(color=BODY_COLORS["torso"]["color"], label='Torso')
    arm_patch = mpatches.Patch(color=BODY_COLORS["arm"]["color"], label='Arms')
    leg_patch = mpatches.Patch(color=BODY_COLORS["leg"]["color"], label='Legs')
    
    ax.legend(handles=[head_patch, torso_patch, arm_patch, leg_patch], loc='upper right')
    return ax

# Create the animation
anim = FuncAnimation(fig, update, frames=len(all_3d_points), interval=100, blit=False)

# Save the animation
# output_animation_path = os.path.join(output_dir, '3d_pose_animation.mp4')
# anim.save(output_animation_path, writer='ffmpeg', fps=10, dpi=200)
anim.save(os.path.join(output_dir, '3d_pose_animation.gif'), writer='pillow', fps=10)

print(f"Animation saved")

# Create a series of side-by-side visualizations for key frames
print("Creating side-by-side visualizations for key frames...")
key_frames = range(0, len(all_3d_points), 10)  # Every 10th frame
for frame in key_frames:
    fig = create_side_by_side_visualization(frame)
    plt.savefig(os.path.join(output_dir, f'side_by_side_frame_{frame*2+350}.png'), dpi=300)
    plt.close(fig)
    print(f"Saved visualization for frame {frame*2+350}")

# Create a final visualization with multiple views
print("Creating final visualization...")
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(2, 3)

# Camera views
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(cv2.cvtColor(all_frames_cam1[-1], cv2.COLOR_BGR2RGB))
ax1.set_title('Final Frame - Camera 233')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(cv2.cvtColor(all_frames_cam2[-1], cv2.COLOR_BGR2RGB))
ax2.set_title('Final Frame - Camera 235')
ax2.axis('off')

# 3D view
ax3 = fig.add_subplot(gs[0, 2], projection='3d')
draw_realistic_human_3d(ax3, all_3d_points[-1], all_landmarks_dicts[-1], all_visibility_dicts[-1])
ax3.set_title('Final 3D Reconstruction')
ax3.set_xlim([996, 1002])
ax3.set_ylim([1004.6, 1006])
ax3.set_zlim([98, 101])

# Trajectory view
ax4 = fig.add_subplot(gs[1, :], projection='3d')
draw_trajectory(ax4, len(all_3d_points)-1, trail_length=len(all_3d_points))
ax4.set_title('Complete Motion Trajectory')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_visualization.png'), dpi=300)
plt.close()

print("Processing complete!")
print(f"All output files saved to {output_dir}")

# Clean up
cv2.destroyAllWindows()
pose.close()
