#!/usr/bin/env python
# coding: utf-8

# In[25]:


import open3d as o3d
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from PIL import Image
import pyheif
import matplotlib.pyplot as plt

# 1. Convert HEIC to PNG for all images in folder
def convert_heic_folder_to_png(heic_folder, png_folder):
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
        
    for file_name in os.listdir(heic_folder):
        if file_name.lower().endswith('.heic'):
            heic_path = os.path.join(heic_folder, file_name)
            output_path = os.path.join(png_folder, os.path.splitext(file_name)[0] + '.png')
            heif_file = pyheif.read(heic_path)
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            image.save(output_path, "PNG")
            print(f"Converted {heic_path} to {output_path}")

# 2. Extract ORB Features from a PNG Image
def extract_image_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    if descriptors is None:
        print(f"No descriptors found in {image_path}")
        return None
    
    return descriptors.flatten()

# 3. Load and Process Point Clouds from PCD Files
def load_point_clouds(pcd_folder):
    pcd_files = [f for f in os.listdir(pcd_folder) if f.endswith('.pcd')]
    point_clouds = []
    for pcd_file in pcd_files:
        pcd_path = os.path.join(pcd_folder, pcd_file)
        pcd = o3d.io.read_point_cloud(pcd_path)
        point_clouds.append(pcd)
    return point_clouds

# 4. Extract FPFH Features from Point Clouds
def extract_fpfh_features(point_clouds):
    fpfh_features = []
    for pcd in point_clouds:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        features = fpfh.data.flatten()
        fpfh_features.append(features)
    return fpfh_features

# 5. Pad/Truncate Features to Ensure Same Length
def pad_or_truncate_features(features_list, target_length=500):
    """
    Pads or truncates feature vectors to ensure they all have the same length.
    """
    padded_features = []
    for features in features_list:
        if len(features) < target_length:
            padded = np.pad(features, (0, target_length - len(features)), 'constant')
        else:
            padded = features[:target_length]
        padded_features.append(padded)
    return padded_features

# 6. Compare Features Using Cosine Similarity
def compare_features(anchor_features, test_features):
    similarity_scores = []
    for feature in test_features:
        similarity = cosine_similarity([anchor_features], [feature])[0][0]
        similarity_scores.append(similarity)
    return similarity_scores

# 7. Visualize Similarity Scores
def visualize_similarity_scores(scores, file_names):
    """
    Visualizes similarity scores as a bar chart.
    The x-axis labels should correspond to the test (point cloud) file names or indices.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(scores)), scores, color='blue')
    plt.xlabel("Point Cloud Index")  # Updated to Point Cloud Index
    plt.ylabel("Similarity Score")
    plt.xticks(range(len(scores)), file_names[:len(scores)], rotation=45, ha="right")  # Adjust labels to match the number of scores
    plt.title("Similarity Scores with Respect to Anchor Images")
    plt.tight_layout()
    plt.show()

# Main Program
if __name__ == "__main__":
    # Folder paths
    heic_folder = '/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/Anchor'
    png_folder = '/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/Anchor_converted'
    pcd_folder = '/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/raw/test/camera_depth_points'
    
    # Step 1: Convert all HEIC images to PNG
    convert_heic_folder_to_png(heic_folder, png_folder)
    
    # Step 2: Load anchor PNG images and extract ORB features
    anchor_features_list = []
    anchor_file_names = []
    
    for file_name in os.listdir(png_folder):
        if file_name.endswith('.png'):
            png_path = os.path.join(png_folder, file_name)
            anchor_features = extract_image_features(png_path)
            if anchor_features is not None:
                anchor_features_list.append(anchor_features)
                anchor_file_names.append(file_name)
    
    if len(anchor_features_list) == 0:
        print("No anchor features were extracted.")
        exit()
    
    # Step 3: Load point clouds from the PCD folder
    point_clouds = load_point_clouds(pcd_folder)
    
    # Step 4: Extract FPFH features from the point clouds
    fpfh_features = extract_fpfh_features(point_clouds)
    
    # Step 5: Pad or truncate both ORB and FPFH features to a consistent length
    target_length = 500  # We are using a target length of 500 for both ORB and FPFH features
    padded_anchor_features = pad_or_truncate_features(anchor_features_list, target_length)
    padded_fpfh_features = pad_or_truncate_features(fpfh_features, target_length)
    
    # Step 6: Compare anchor features with point cloud features
    for i, anchor_features in enumerate(padded_anchor_features):
        similarity_scores = compare_features(anchor_features, padded_fpfh_features)
        print(f"Similarity Scores for Anchor Image {anchor_file_names[i]}:", similarity_scores)
        
        # Use point cloud file names for visualization labels instead of anchor file names
        point_cloud_file_names = [f'Point Cloud {i+1}' for i in range(len(similarity_scores))]
        
        # Visualize Similarity Scores for each anchor image
        visualize_similarity_scores(similarity_scores, point_cloud_file_names)


# In[ ]:




