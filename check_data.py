import json
import os
import numpy as np

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

def main():
    # Load 3D points data
    print("Loading 3D points data...")
    points_data = load_3d_points_data()
    
    if points_data:
        print(f"Data type: {type(points_data)}")
        
        # Check if it's a dictionary
        if isinstance(points_data, dict):
            print("Data is a dictionary with keys:")
            print(list(points_data.keys())[:5])  # Show first 5 keys
            
            # Print a sample entry
            first_key = list(points_data.keys())[0]
            print(f"\nSample entry for key '{first_key}':")
            print(f"Type: {type(points_data[first_key])}")
            print(f"Length: {len(points_data[first_key])}")
            if len(points_data[first_key]) > 0:
                print(f"First item type: {type(points_data[first_key][0])}")
                print(f"Example: {points_data[first_key][0]}")
        
        # Check if it's a list
        elif isinstance(points_data, list):
            print("Data is a list with length:", len(points_data))
            
            # Print a sample entry
            if len(points_data) > 0:
                print(f"\nSample entry for index 0:")
                print(f"Type: {type(points_data[0])}")
                if isinstance(points_data[0], list):
                    print(f"Length: {len(points_data[0])}")
                    if len(points_data[0]) > 0:
                        print(f"First item type: {type(points_data[0][0])}")
                        print(f"Example: {points_data[0][0]}")
        
        # General output of the first item, whatever it is
        print("\nRaw sample:")
        print(points_data)
    else:
        print("Failed to load 3D points data.")

if __name__ == "__main__":
    main() 