#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# 3. Extract SIFT Features using mask to ignore padded area
def extract_sift_features(image, mask=None):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, mask)
    return keypoints, descriptors

# 4. Calculate Similarity Score using match ratio
def match_features(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.75
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    # Similarity score based on match ratio
    if len(matches) > 0:
        similarity_score = len(good_matches) / len(matches)
    else:
        similarity_score = 0

    return good_matches, similarity_score

# 5. Retrieve Best Matches with similarity score
def retrieve_best_matches(anchor_image_path, dataset_image_paths):
    anchor_image = cv2.imread(anchor_image_path)
    gray_anchor = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
    kp1, des1 = extract_sift_features(gray_anchor)
    if des1 is None:
        print("No features found in anchor image.")
        return []

    results = []
    for image_path in dataset_image_paths:
        dataset_image = cv2.imread(image_path)
        gray_dataset = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2GRAY)
        kp2, des2 = extract_sift_features(gray_dataset)
        if des2 is None:
            continue
        good_matches, similarity_score = match_features(des1, des2)
        results.append((image_path, similarity_score, kp1, kp2, good_matches))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# 6. Visualize Similarity Scores
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

# 7. Draw Matches (With Feature Keypoints Visualization)
def draw_matches(img1_path, img2_path, kp1, kp2, matches):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Visualize keypoints detected for debugging
    img1_keypoints = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_keypoints = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Anchor Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Dataset Image')
    plt.show()

    # Ensure matches and keypoints are sorted and compatible
    good_matches = sorted(matches, key=lambda x: x.distance)

    if len(good_matches) > 0:
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Top {len(good_matches)} SIFT Matches")
        plt.axis('off')
        plt.show()

# 8. Main Function
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
        results = retrieve_best_matches(anchor_image_path, dataset_image_paths)
        if not results:
            print("No matching images found.")
            continue

        top_n = 5
        print(f"Top {top_n} matches:")
        for i in range(min(top_n, len(results))):
            image_path, score, kp1, kp2, good_matches = results[i]
            print(f"{i+1}. Image: {image_path}, Similarity Score: {score:.2f}")

        visualize_similarity_scores(results, top_n, anchor_image_path)
        top_match_image_path = results[0][0]
        kp1, kp2, good_matches = results[0][2], results[0][3], results[0][4]
        if kp1 and kp2:
            draw_matches(anchor_image_path, top_match_image_path, kp1, kp2, good_matches)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




