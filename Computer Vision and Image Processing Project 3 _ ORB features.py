#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import pyheif
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# 1. Convert HEIC to PNG
def convert_heic_to_png(heic_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for file_name in os.listdir(heic_folder):
        if file_name.lower().endswith('.heic'):
            heic_path = os.path.join(heic_folder, file_name)
            try:
                heif_file = pyheif.read(heic_path)
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                output_file_name = os.path.splitext(file_name)[0] + '.png'
                output_path = os.path.join(output_folder, output_file_name)
                image.save(output_path, "PNG")
                print(f"Converted {file_name} to {output_file_name}")
            except Exception as e:
                print(f"Error converting {file_name}: {e}")

# 2. Get Dataset Image and PCD Paths
def get_dataset_paths(raw_folder, image_folders, pcd_folders):
    image_paths = []
    for image_folder in image_folders:
        folder_path = os.path.join(raw_folder, image_folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, file)
                image_paths.append(image_path)
    
    # Get PCD files
    pcd_paths = []
    for pcd_folder in pcd_folders:
        pcd_folder_path = os.path.join(raw_folder, pcd_folder)
        if not os.path.exists(pcd_folder_path):
            print(f"Warning: Folder {pcd_folder_path} does not exist.")
            continue
        for file in os.listdir(pcd_folder_path):
            if file.lower().endswith('.pcd'):
                pcd_path = os.path.join(pcd_folder_path, file)
                pcd_paths.append(pcd_path)
    return image_paths, pcd_paths

# 3. Resize Image with Aspect Ratio
def resize_image_with_aspect_ratio(image, target_size, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / float(w), target_height / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=inter)

# 4. Pad Image to Target Size
def pad_image_to_size(image, target_size):
    h, w = image.shape[:2]
    delta_w, delta_h = target_size[0] - w, target_size[1] - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# 5. Extract Color Histogram (for Image Features)
def extract_color_histogram(image, bins=256):
    if len(image.shape) == 2 or image.shape[2] == 1:
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 6. Extract ORB Features (for Image Features)
def extract_orb_features(image, max_features=500):
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# 7. Extract Combined Features (ORB + Histogram) for Images
def extract_features(image_path, target_size=None, max_features=500):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to load image {image_path}")
        return None, None

    if target_size is not None:
        resized_image = resize_image_with_aspect_ratio(image, target_size)
        padded_image = pad_image_to_size(resized_image, target_size)
    else:
        padded_image = image

    gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
    color_hist = extract_color_histogram(padded_image)
    keypoints, descriptors = extract_orb_features(gray_image, max_features=max_features)

    descriptor_length = 32
    if descriptors is None:
        descriptors = np.zeros((max_features, descriptor_length), dtype=np.uint8)
    if descriptors.shape[0] < max_features:
        padding = np.zeros((max_features - descriptors.shape[0], descriptor_length), dtype=np.uint8)
        descriptors = np.vstack((descriptors, padding))
    descriptors = descriptors.flatten()
    combined_features = np.hstack((color_hist, descriptors))
    return keypoints, combined_features

# 8. Calculate Similarity Score (Image Feature Matching)
def calculate_similarity_score(feature1, feature2):
    return cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))[0][0]

# 9. Analyze PCD Files and Extract FPFH Features
def analyze_pcd_files(pcd_paths):
    pcd_features = []
    for pcd_path in pcd_paths:
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.estimate_normals()
            radius_feature = 0.1
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            features = fpfh.data.flatten()
            pcd_features.append((pcd_path, features))
        except Exception as e:
            print(f"Error processing PCD file {pcd_path}: {e}")
    return pcd_features

# 10. Retrieve Best Matches (Image)
def retrieve_best_matches(anchor_image_path, dataset_image_paths, target_size, max_features):
    kp1, features1 = extract_features(anchor_image_path, target_size=target_size, max_features=max_features)
    if features1 is None:
        return []

    results = []
    for image_path in dataset_image_paths:
        if 'camera_color_image_raw' not in image_path:
            continue
        kp2, features2 = extract_features(image_path, target_size=target_size, max_features=max_features)
        if features2 is None:
            continue
        score = calculate_similarity_score(features1, features2)
        results.append((image_path, kp1, kp2, score))
    results.sort(key=lambda x: x[3], reverse=True)
    return results

# 11. Visualize Matches Using OpenCV
def visualize_matches(anchor_image_path, best_match_image_path, kp1, kp2, target_size):
    img1 = cv2.imread(anchor_image_path)
    img2 = cv2.imread(best_match_image_path)

    # Resize images for better visualization
    img1_resized = resize_image_with_aspect_ratio(img1, target_size)
    img1_padded = pad_image_to_size(img1_resized, target_size)
    img2_resized = resize_image_with_aspect_ratio(img2, target_size)
    img2_padded = pad_image_to_size(img2_resized, target_size)

    gray1 = cv2.cvtColor(img1_padded, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_padded, cv2.COLOR_BGR2GRAY)

    kp1, des1 = extract_orb_features(gray1)
    kp2, des2 = extract_orb_features(gray2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(img1_padded, kp1, img2_padded, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Keypoint Matches Between {os.path.basename(anchor_image_path)} and {os.path.basename(best_match_image_path)}")
    plt.axis('off')
    plt.show()

# 12. Visualize Similarity Scores
def visualize_similarity_scores(results, top_n=5):
    top_results = results[:top_n]
    image_names = [os.path.basename(res[0]) for res in top_results]
    scores = [res[3] for res in top_results]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(scores)), scores, color='blue')
    plt.yticks(range(len(image_names)), image_names)
    plt.title("Top Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.gca().invert_yaxis()
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{score:.4f}", va='center')

    plt.show()

# 13. Main Function
def main():
    TARGET_SIZE = (640, 400)
    MAX_FEATURES = 500

    # Paths to your data folders (Update these paths)
    anchor_heic_folder = "/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/Anchor"  # Folder containing .heic anchor images
    anchor_converted_folder = "/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/Anchor_converted"  # Folder to save converted .png images
    raw_folder = "/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/raw/test"

    image_folders = ["camera_color_image_raw"]
    pcd_folders = ["camera_depth_points"]

    # Step 1: Convert HEIC to PNG
    convert_heic_to_png(heic_folder=anchor_heic_folder, output_folder=anchor_converted_folder)

    # Step 2: Get Anchor Images and Dataset Image Paths
    anchor_images = [os.path.join(anchor_converted_folder, f) for f in os.listdir(anchor_converted_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_image_paths, pcd_paths = get_dataset_paths(raw_folder, image_folders, pcd_folders)

    for anchor_image_path in anchor_images:
        print(f"\nProcessing Anchor Image: {anchor_image_path}")
        
        # Step 3: Retrieve Best Matches
        results = retrieve_best_matches(anchor_image_path, dataset_image_paths, target_size=TARGET_SIZE, max_features=MAX_FEATURES)
        if results:
            best_match_image_path = results[0][0]
            print(f"Best match for {os.path.basename(anchor_image_path)}: {best_match_image_path} with similarity score {results[0][3]:.4f}")
            
            # Step 4: Visualize Matches
            visualize_matches(anchor_image_path, best_match_image_path, results[0][1], results[0][2], target_size=TARGET_SIZE)
            
            # Step 5: Visualize Similarity Scores
            visualize_similarity_scores(results, top_n=5)
        else:
            print(f"No matches found for {anchor_image_path}")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




