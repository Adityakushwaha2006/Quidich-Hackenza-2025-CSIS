import cv2
import mediapipe as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial

# Configuration
NUM_CORES = mp.cpu_count()
CHUNK_SIZE = 10  # Adjust based on memory constraints
FRAME_START, FRAME_END = 1, 506

# Paths
c233_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\233_im'
c235_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\235_im'
intrinsic_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\intrinsic.json'
extrinsic_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\extrinsic.json'

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

def init_worker():
    global pose
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        use_gpu=True  # Enable GPU acceleration
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

def process_frame_batch(frame_batch):
    batch_results = []
    for frame_num in frame_batch:
        file1 = f"HPUP_033_1_1_1_L_CAM-05_{frame_num:07d}.jpeg"
        file2 = f"HPUP_033_1_1_1_L_CAM-02_{frame_num:07d}.jpeg"
        
        frame1 = cv2.imread(os.path.join(c233_path, file1))
        frame2 = cv2.imread(os.path.join(c235_path, file2))
        
        if frame1 is None or frame2 is None:
            continue

        results1 = pose.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        results2 = pose.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        points_3d = []
        if results1.pose_landmarks and results2.pose_landmarks:
            for i in range(33):
                point1 = np.array([results1.pose_landmarks.landmark[i].x * frame1.shape[1],
                                   results1.pose_landmarks.landmark[i].y * frame1.shape[0]])
                point2 = np.array([results2.pose_landmarks.landmark[i].x * frame2.shape[1],
                                   results2.pose_landmarks.landmark[i].y * frame2.shape[0]])
                
                point_3d = triangulate_point(P1, P2, point1, point2)
                points_3d.append(point_3d)

        batch_results.append(np.array(points_3d))
    return batch_results

def main():
    frame_numbers = list(range(FRAME_START, FRAME_END + 1))
    batches = [frame_numbers[i:i+CHUNK_SIZE] 
              for i in range(0, len(frame_numbers), CHUNK_SIZE)]

    with Manager() as manager:
        all_points = manager.list()
        
        with Pool(
            processes=NUM_CORES,
            initializer=init_worker,
            maxtasksperchild=10
        ) as pool:
            results = pool.imap_unordered(process_frame_batch, batches)
            
            for i, batch_result in enumerate(results):
                all_points.extend(batch_result)
                print(f"Processed batch {i+1}/{len(batches)}")

        final_points = list(all_points)

    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.jet(np.linspace(0, 1, len(final_points)))

    for i, points in enumerate(final_points):
        if points.ndim == 2 and points.shape[1] == 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[colors[i]], s=5)
        else:
            print(f"Skipping invalid points array at index {i}: {points}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Estimation')
    plt.show()

if __name__ == "__main__":
    main()
    print("MediaPipe using GPU:", pose._gpu_initialized)
