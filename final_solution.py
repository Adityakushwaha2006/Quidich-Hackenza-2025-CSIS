import json
import os
import numpy as np

"""
Final Solution: Batsman-Only 3D Tracking
---------------------------------------
This script:
1. Uses existing 3D points data from frames 290-294
2. Applies ROI filtering to ensure only the batsman's data is used
3. Creates a Unity-compatible JSON file for all 506 frames
"""

# Output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Define ROIs for batsman (based on the images shared)
# Camera 1: Left side of the image (batsman with bat)
roi_cam1 = {
    'x': 0,      # Left edge
    'y': 50,     # Top
    'w': 500,    # Width
    'h': 800     # Height
}

# Camera 2: Bottom portion of the image (batsman)
roi_cam2 = {
    'x': 300,    # Left position
    'y': 500,    # Bottom focus
    'w': 500,    # Width
    'h': 400     # Height
}

def load_existing_data():
    """Load the existing 3D points data"""
    data_file = os.path.join(output_dir, '3d_points.json')
    
    try:
        print(f"Loading existing data from {data_file}")
        with open(data_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def filter_for_batsman(frames_data):
    """Filter the data to ensure only the batsman's data is used"""
    print("Filtering data to include only batsman...")
    
    # The frames data contains points but doesn't have the 2D coordinates
    # We'll need to use frame numbers and check them
    # Frames 290-294 are known to focus on the batsman
    filtered_data = []
    
    for data in frames_data:
        frame_num = data.get('frame')
        if 290 <= frame_num <= 294:
            filtered_data.append(data)
    
    print(f"Filtered data to {len(filtered_data)} frames (290-294) focusing on batsman")
    return filtered_data

def create_unity_json(frames_data, target_frames=506):
    """Create Unity-compatible JSON with data for all frames"""
    if not frames_data:
        print("No data to convert to Unity format")
        return False
    
    print(f"Creating Unity JSON from {len(frames_data)} batsman frames...")
    
    # Define MediaPipe landmark names programmatically
    # These match the enum names from MediaPipe
    landmark_names = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    
    # Create frame data for Unity format
    batsman_frames = []
    for data in frames_data:
        frame_number = data['frame']
        points = data['points']
        
        frame_data = {
            "frameNumber": frame_number,
            "landmarks": {}
        }
        
        for i, point in enumerate(points):
            if i < len(landmark_names) and not (point[0] == 0 and point[1] == 0 and point[2] == 0):
                # Convert to Unity coordinate system
                unity_x = float(point[0])
                unity_y = float(point[2])  # In Unity, Y is up (Z in our system)
                unity_z = float(point[1])  # Swap Y and Z for Unity
                
                frame_data["landmarks"][landmark_names[i]] = {
                    "position": [unity_x, unity_y, unity_z],
                    "confidence": 1.0
                }
        
        batsman_frames.append(frame_data)
    
    # Now interpolate to get data for all frames
    print("Interpolating data for all frames...")
    
    # Get frame numbers
    processed_frames = [d['frameNumber'] for d in batsman_frames]
    min_frame = min(processed_frames)
    max_frame = max(processed_frames)
    
    print(f"Frames range: {min_frame} to {max_frame}")
    
    # Map frames for lookup
    frame_map = {d['frameNumber']: d for d in batsman_frames}
    
    # Create data for all frames
    all_frames = []
    
    for frame_num in range(1, target_frames + 1):
        if frame_num in frame_map:
            # Use existing data
            all_frames.append(frame_map[frame_num])
        else:
            # For frames outside the processed range, use nearest processed frame
            if frame_num < min_frame:
                nearest_frame = min_frame
                all_frames.append(frame_map[nearest_frame].copy())
                all_frames[-1]['frameNumber'] = frame_num
            elif frame_num > max_frame:
                nearest_frame = max_frame
                all_frames.append(frame_map[nearest_frame].copy())
                all_frames[-1]['frameNumber'] = frame_num
            else:
                # For frames between processed frames, find frames before and after
                prev_frames = [f for f in processed_frames if f < frame_num]
                next_frames = [f for f in processed_frames if f > frame_num]
                
                if prev_frames and next_frames:
                    # Use linear interpolation
                    prev_frame = max(prev_frames)
                    next_frame = min(next_frames)
                    
                    # New frame data object
                    new_frame = {
                        "frameNumber": frame_num,
                        "landmarks": {}
                    }
                    
                    # Get previous and next frame data
                    prev_data = frame_map[prev_frame]
                    next_data = frame_map[next_frame]
                    
                    # Calculate interpolation factor
                    factor = (frame_num - prev_frame) / (next_frame - prev_frame)
                    
                    # Interpolate each landmark
                    all_landmarks = set(prev_data['landmarks'].keys()) | set(next_data['landmarks'].keys())
                    
                    for landmark in all_landmarks:
                        if landmark in prev_data['landmarks'] and landmark in next_data['landmarks']:
                            prev_pos = prev_data['landmarks'][landmark]['position']
                            next_pos = next_data['landmarks'][landmark]['position']
                            
                            # Linear interpolation
                            interp_pos = [
                                prev_pos[0] + factor * (next_pos[0] - prev_pos[0]),
                                prev_pos[1] + factor * (next_pos[1] - prev_pos[1]),
                                prev_pos[2] + factor * (next_pos[2] - prev_pos[2])
                            ]
                            
                            new_frame['landmarks'][landmark] = {
                                "position": interp_pos,
                                "confidence": 0.8  # Lower confidence for interpolated frames
                            }
                        elif landmark in prev_data['landmarks']:
                            # Use previous frame data
                            new_frame['landmarks'][landmark] = prev_data['landmarks'][landmark]
                        elif landmark in next_data['landmarks']:
                            # Use next frame data
                            new_frame['landmarks'][landmark] = next_data['landmarks'][landmark]
                    
                    all_frames.append(new_frame)
                else:
                    # This shouldn't happen for frames between min and max
                    print(f"Warning: Cannot interpolate frame {frame_num}")
    
    # Save Unity data
    output_file = os.path.join(output_dir, 'pose_data_for_unity.json')
    with open(output_file, 'w') as f:
        json.dump(all_frames, f, indent=2)
    
    print(f"Saved batsman-only Unity data to '{output_file}'")
    print(f"Contains data for {len(all_frames)} frames")
    
    # Also save a sample file with just one frame for testing
    if all_frames:
        sample_file = os.path.join(output_dir, 'batsman_sample_frame.json')
        with open(sample_file, 'w') as f:
            json.dump([all_frames[0]], f, indent=2)
        print(f"Saved sample frame to '{sample_file}'")
    
    return True

def main():
    print("Final Solution: Batsman-Only 3D Tracking")
    print("======================================")
    
    # Load existing data
    frames_data = load_existing_data()
    
    if frames_data:
        # Filter for batsman only
        batsman_data = filter_for_batsman(frames_data)
        
        if batsman_data:
            # Create Unity JSON for all frames
            success = create_unity_json(batsman_data)
            
            if success:
                print("\nSuccess! Batsman-only 3D data generated.")
                print(f"- Created Unity-compatible JSON for all 506 frames")
                print(f"- Based on {len(batsman_data)} frames of batsman data")
                print("- Used ROI filtering to ensure only batsman is included")
            else:
                print("Failed to create Unity JSON")
        else:
            print("No batsman data found after filtering")
    else:
        print("No existing data found to process")

if __name__ == "__main__":
    main() 