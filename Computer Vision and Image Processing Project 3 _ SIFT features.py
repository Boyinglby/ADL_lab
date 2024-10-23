#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
from PIL import Image
import pyheif
import matplotlib.pyplot as plt

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

# 2. Get Dataset Image Paths
def get_dataset_image_paths(raw_folder):
    image_paths = []
    for root, dirs, files in os.walk(raw_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'depth' not in file.lower():
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths

# 3. Resize image to a consistent size
def resize_image(image, target_size=(640, 480)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

# 4. Segment the fire extinguisher based on color
def segment_fire_extinguisher(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image, mask

# 5. Extract SIFT Features using mask to ignore padded area
def extract_sift_features(image, mask=None):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, mask)
    return keypoints, descriptors

# 6. Calculate Similarity Score using match ratio
def match_features(des1, des2):
    # Ensure descriptors are not empty and have enough data
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], 0  # Return empty matches and a zero similarity score

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.75
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(matches) > 0:
        similarity_score = len(good_matches) / len(matches)
    else:
        similarity_score = 0

    return good_matches, similarity_score

# 7. Retrieve Best Matches with similarity score using segmentation
def retrieve_best_matches(anchor_image_path, dataset_image_paths, target_size=(640, 480)):
    anchor_image = cv2.imread(anchor_image_path)
    
    # Resize the anchor image first
    resized_anchor = resize_image(anchor_image, target_size)
    
    # Segment fire extinguisher from the anchor image
    segmented_anchor, mask_anchor = segment_fire_extinguisher(resized_anchor)
    gray_anchor = cv2.cvtColor(segmented_anchor, cv2.COLOR_BGR2GRAY)
    kp1, des1 = extract_sift_features(gray_anchor, mask_anchor)
    
    if des1 is None:
        print("No features found in anchor image.")
        return []

    results = []
    for image_path in dataset_image_paths:
        dataset_image = cv2.imread(image_path)
        
        # Resize the dataset image
        resized_dataset = resize_image(dataset_image, target_size)
        
        # Segment fire extinguisher from the dataset image
        segmented_dataset, mask_dataset = segment_fire_extinguisher(resized_dataset)
        gray_dataset = cv2.cvtColor(segmented_dataset, cv2.COLOR_BGR2GRAY)
        kp2, des2 = extract_sift_features(gray_dataset, mask_dataset)
        
        # Check if features were found before proceeding with matching
        if des2 is None:
            continue
        good_matches, similarity_score = match_features(des1, des2)
        results.append((image_path, similarity_score, kp1, kp2, good_matches, segmented_anchor, segmented_dataset))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# 8. Visualize Similarity Scores
def visualize_similarity_scores(results, top_n, anchor_image_path):
    image_names = [os.path.basename(result[0]) for result in results[:top_n]]
    scores = [result[1] for result in results[:top_n]]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(scores)), scores, color='blue')
    plt.title(f"Top {top_n} Similarity Scores for {os.path.basename(anchor_image_path)}")
    plt.xlabel("Image")
    plt.ylabel("Similarity Score (Match Ratio)")
    plt.xticks(range(len(scores)), image_names, rotation=45, ha='right')
    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{score:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

# 9. Draw Matches (With Feature Keypoints Visualization)
def draw_matches(results):
    anchor_image_path, similarity_score, kp1, kp2, good_matches, segmented_anchor, segmented_dataset = results[0]
    
    img1_keypoints = cv2.drawKeypoints(segmented_anchor, kp1, None, color=(0, 255, 0))
    img2_keypoints = cv2.drawKeypoints(segmented_dataset, kp2, None, color=(0, 255, 0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Anchor Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Dataset Image')
    plt.show()

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    if len(good_matches) > 0:
        img_matches = cv2.drawMatches(
            segmented_anchor, kp1, segmented_dataset, kp2, good_matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Top {len(good_matches)} SIFT Matches")
        plt.axis('off')
        plt.show()

# 10. Main Function
def main():
    # Paths to your data folders (Update these paths)
    anchor_heic_folder = "/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/Anchor"  # Folder containing .heic anchor images
    anchor_converted_folder = "/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/Anchor_converted"  # Folder to save converted .png images
    raw_folder = "/Users/lianhechu/Documents/Applied AI Master Program/R7020E Computer vision and image processing/Lab/Projects/Project_3/raw/test"  # Folder containing the raw dataset images

    convert_heic_to_png(heic_folder=anchor_heic_folder, output_folder=anchor_converted_folder)
    anchor_images = [os.path.join(anchor_converted_folder, f) for f in os.listdir(anchor_converted_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not anchor_images:
        print("No anchor images found.")
        return

    dataset_image_paths = get_dataset_image_paths(raw_folder)
    if not dataset_image_paths:
        print("No dataset images found.")
        return

    print(f"Total dataset images: {len(dataset_image_paths)}")
    for anchor_image_path in anchor_images:
        print(f"\nProcessing Anchor Image: {anchor_image_path}")
        results = retrieve_best_matches(anchor_image_path, dataset_image_paths, target_size=(640, 480))
        if not results:
            print("No matching images found.")
            continue

        top_n = 5
        print(f"Top {top_n} matches:")
        for i in range(min(top_n, len(results))):
            image_path, score, kp1, kp2, good_matches, _, _ = results[i]
            print(f"{i+1}. Image: {image_path}, Similarity Score: {score:.2f}")

        visualize_similarity_scores(results, top_n, anchor_image_path)
        draw_matches(results)

if __name__ == "__main__":
    main()
