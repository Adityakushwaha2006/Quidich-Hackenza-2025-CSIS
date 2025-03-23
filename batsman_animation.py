import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

# Define the connections between landmarks for skeleton visualization
POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # Arms
    (0, 7), (7, 8), (8, 9), (9, 10),  # Right leg
    (0, 11), (11, 12), (12, 13), (13, 14),  # Left leg
    (0, 15), (15, 16),  # Shoulders to ears
    (15, 17), (16, 18),  # Ears to eyes
    (17, 19), (18, 20),  # Eyes to nose
    (19, 20), (11, 7)  # Nose to mouth, hips
]

def load_3d_points_data(json_file='output/3d_points.json'):
    """Load 3D points data from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {json_file} is not valid JSON.")
        return None

def create_batsman_animation(frames_data, output_gif='output/batsman_skeleton.gif', output_static='output/batsman_static.png'):
    """Create an animation of the batsman skeleton"""
    if not frames_data:
        print("No frame data available for animation.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    
    # Get all frame indices
    frame_indices = sorted(list(frames_data.keys()))
    
    # Sample frames for animation (to reduce file size)
    sample_rate = max(1, len(frame_indices) // 50)  # Limit to ~50 frames
    sampled_indices = frame_indices[::sample_rate]
    
    if len(sampled_indices) == 0:
        print("No frames to animate after sampling.")
        return
    
    # Create a static figure of the first frame
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Function to draw a single frame
    def draw_skeleton(ax, landmarks):
        ax.clear()
        
        # Plot points
        ax.scatter(landmarks[:, 0], landmarks[:, 2], landmarks[:, 1], c='blue', s=20)
        
        # Plot the skeleton connections
        for connection in POSE_CONNECTIONS:
            if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                ax.plot([landmarks[connection[0], 0], landmarks[connection[1], 0]],
                        [landmarks[connection[0], 2], landmarks[connection[1], 2]],
                        [landmarks[connection[0], 1], landmarks[connection[1], 1]], 'r-')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Z') 
        ax.set_zlabel('Y')
        ax.set_title('Batsman 3D Pose')
        
        # Set consistent axis limits for all frames
        all_points = np.vstack([frames_data[idx] for idx in frame_indices])
        x_range = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))
        y_range = (np.min(all_points[:, 1]), np.max(all_points[:, 1]))
        z_range = (np.min(all_points[:, 2]), np.max(all_points[:, 2]))
        
        # Set limits with some padding
        padding = 0.1
        x_padding = padding * (x_range[1] - x_range[0])
        y_padding = padding * (y_range[1] - y_range[0])
        z_padding = padding * (z_range[1] - z_range[0])
        
        ax.set_xlim(x_range[0] - x_padding, x_range[1] + x_padding)
        ax.set_ylim(z_range[0] - z_padding, z_range[1] + z_padding)
        ax.set_zlim(y_range[0] - y_padding, y_range[1] + y_padding)
        
    # Draw static image for first frame
    first_frame = frames_data[frame_indices[0]]
    draw_skeleton(ax, first_frame)
    plt.savefig(output_static)
    plt.close()
    
    # Create animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        landmarks = frames_data[frame_idx]
        draw_skeleton(ax, landmarks)
        return ax,
    
    ani = FuncAnimation(fig, update, frames=sampled_indices, interval=100, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=10)
    ani.save(output_gif, writer=writer)
    plt.close()
    
    print(f"Static visualization saved to {output_static}")
    print(f"Animation saved to {output_gif}")

def interpolate_for_all_frames(batsman_frames, total_frames=506):
    """Interpolate batsman data for all frames"""
    # Get the source frame indices (frames we actually have data for)
    source_frames = sorted(list(batsman_frames.keys()))
    
    if len(source_frames) == 0:
        return {}
    
    # Create output dictionary for all frames
    all_frames = {}
    
    # Get the shape of landmark data
    landmark_shape = batsman_frames[source_frames[0]].shape
    
    # For frames before the first known frame, use the first frame's data
    for frame_idx in range(1, source_frames[0]):
        all_frames[frame_idx] = batsman_frames[source_frames[0]].copy()
    
    # For frames after the last known frame, use the last frame's data
    for frame_idx in range(source_frames[-1] + 1, total_frames + 1):
        all_frames[frame_idx] = batsman_frames[source_frames[-1]].copy()
    
    # Copy the known frames
    for frame_idx in source_frames:
        all_frames[frame_idx] = batsman_frames[frame_idx].copy()
    
    # Interpolate between known frames
    for i in range(len(source_frames) - 1):
        start_frame = source_frames[i]
        end_frame = source_frames[i + 1]
        start_data = batsman_frames[start_frame]
        end_data = batsman_frames[end_frame]
        
        for frame_idx in range(start_frame + 1, end_frame):
            # Calculate interpolation factor (0 to 1)
            t = (frame_idx - start_frame) / (end_frame - start_frame)
            # Linear interpolation for each landmark
            all_frames[frame_idx] = start_data + t * (end_data - start_data)
    
    return all_frames

def main():
    # Define ROI for batsman (same as in final_solution.py)
    # These are the coordinates to isolate the batsman in each camera view
    roi_cam1 = {
        'x_min': 900, 'x_max': 1100, 
        'y_min': 400, 'y_max': 700
    }
    
    roi_cam2 = {
        'x_min': 800, 'x_max': 1000, 
        'y_min': 400, 'y_max': 700
    }
    
    # Load 3D points data
    print("Loading 3D points data...")
    points_data = load_3d_points_data()
    
    if points_data:
        print(f"Successfully loaded data with {len(points_data)} frames")
        
        # Check the type of points_data
        if isinstance(points_data, list):
            # Convert list to dictionary with frame index as key
            frames_data = {}
            for i, landmarks in enumerate(points_data):
                if landmarks:  # Skip empty frames
                    frames_data[i + 290] = np.array(landmarks)
        else:
            # Handle dictionary format
            frames_data = {int(frame_idx): np.array(landmarks) 
                          for frame_idx, landmarks in points_data.items()
                          if landmarks}  # Skip empty frames
        
        # Filter for frames that contain the batsman (frames 290-294)
        batsman_frames = {idx: landmarks for idx, landmarks in frames_data.items() 
                         if 290 <= idx <= 294}
        
        if batsman_frames:
            print(f"Found {len(batsman_frames)} frames with batsman data")
            
            # Interpolate data for all 506 frames
            print("Interpolating data for all 506 frames...")
            all_frames = interpolate_for_all_frames(batsman_frames, 506)
            print(f"Created data for {len(all_frames)} frames")
            
            # Create animation
            print("Creating batsman animation for all frames...")
            create_batsman_animation(all_frames, 
                                     output_gif='output/batsman_all_frames.gif',
                                     output_static='output/batsman_all_frames_static.png')
            
            print("Batsman animation complete!")
        else:
            print("No batsman frames found.")
    else:
        print("Failed to load 3D points data.")

if __name__ == "__main__":
    main() 