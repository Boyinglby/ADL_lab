import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
import pandas as pd

CAMERAINFO_RGB_PATH = './test/camera_color_camera_info'
CAMERAINFO_depth_PATH = './test/camera_depth_camera_info'

RAW_RGB_PATH = './test/camera_color_image_raw'
RAW_depth_PATH = './test/camera_depth_image_raw'

ANCHOR_PATH = './anchor' #  png anchor images

def read_camera_info(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract timestamp information
    secs = re.search(r'secs: \s*(\d+)', content).group(1)
    
    nsecs = re.search(r'nsecs: \s*(\d+)', content).group(1)
    timestamp = str(secs+nsecs) 
    
    # Extract camera matrix (K)
    K_values = re.search(r'K: \[(.*?)\]', content).group(1)
    K = [float(x) for x in K_values.split(', ')]
    K = np.array(K).reshape((3, 3))
    
    # Extract distortion coefficients (D)
    D_values = re.search(r'D: \[(.*?)\]', content).group(1)
    D = [float(x) for x in D_values.split(', ')]
    
    return timestamp, K, D

def timestampAll(folder_path):
    # Loop over all files in the folder
    timestampAll = []
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            
            timestamp = read_camera_info(full_path)[0]
            if len(timestamp)!=19:
                timestamp += '0'
            timestamp = int(timestamp)
            timestampAll.append(timestamp)
            
    return timestampAll

def find_closest_timestamp(target, timestamps):
    return min(timestamps, key=lambda x: abs(x - target))

def synchronize_images(rgb_timestamps, depth_timestamps, rgb_images, depth_images):
    synchronized_pairs = []
    for rgb_time in rgb_timestamps:
        closest_depth_time = find_closest_timestamp(rgb_time, depth_timestamps)
        rgb_idx = rgb_timestamps.index(rgb_time)
        depth_idx = depth_timestamps.index(closest_depth_time)
        synchronized_pairs.append((rgb_images[rgb_idx], depth_images[depth_idx]))
    return synchronized_pairs

def edgedepth(depth_image):
    # Normalize the depth image
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalized = np.uint8(depth_image_normalized)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(depth_image_normalized)

    # Apply Gaussian blur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(clahe_image, (7, 7), 3)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    return edges

     
def extract_sift_features(image, mask=None):
    # as anchor image has high resolution contains a lot of details
    # so here we check the large scale keypoints
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    large_scale_keypoints = [kp for kp in keypoints if kp.size > 10]
    large_scale_descriptors = [descriptors[i] for i, kp in enumerate(keypoints) if kp.size > 10]
    
    return large_scale_keypoints, np.array(large_scale_descriptors)



def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    try:
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
    except:
        pass
    
    return good_matches, len(good_matches)/len(matches)


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

def segmented_match_score(anchor, rgb):
    score = 0
    rgb_seg, _ = segment_fire_extinguisher(rgb)
    anchor_seg, _ = segment_fire_extinguisher(anchor)
    
    anchor_seg_kp, anchor_seg_des = extract_sift_features(anchor_seg)
    rgb_seg_kp, rgb_seg_des = extract_sift_features(rgb_seg)
    
    if rgb_seg_kp:
        matches,score = match_features(anchor_seg_des, rgb_seg_des)   
    
        print(f"Match Score of red extinguisher: {score}")      
             
    return score

def depth_match_score(anchor, depth):
    score = 0
    blurred_depth = cv2.GaussianBlur(np.array(depth),(5,5),5)
    depth_edge = edgedepth(blurred_depth)
    
    blurred_anchor = cv2.GaussianBlur(np.array(anchor),(5,5),5)
    anchor_edge = cv2.Canny(blurred_anchor, threshold1=50, threshold2=150)

    anchor_edge_kp, anchor_edge_des = extract_sift_features(anchor_edge)

    depth_edge_kp, depth_edge_des = extract_sift_features(depth_edge)
    
    if depth_edge_kp:
        matches,score = match_features(anchor_edge_des, depth_edge_des)   

        print(f"Match Score of depth image: {score}")      
             
    return score
    
def retrieve_objects(anchor, rgb, depth, weight=0.1):
    
    # combine depth similarity and rgb red area similarity with adjustable weight
    
    depth_score = depth_match_score(anchor, depth)
    red_score = segmented_match_score(anchor, rgb)
    score_sum = depth_score + weight*red_score
              
    return score_sum

def visualize_score(scores, anchor_name):
    # visulaize score bar plot

    x_tick = range(1, len(scores) + 1)

    # Create a DataFrame
    data = pd.DataFrame({'Image': x_tick, 'Score': scores})

    # Create the bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Image', y='Score', data=data)

    # Customizing the plot
    plt.title(f"Similarity Scores of anchor{anchor_name}")
    plt.xlabel("Image")
    plt.ylabel("Similarity Score (Match Ratio)")
    plt.xticks(rotation=45, fontsize=5)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

#----------------Threshold not set yet----------------
# Retrieve objects with scores above a certain threshold
threshold = 0


#--------------------sync rgb and depth pairs-------------------------      
 
rgb_timestamps = timestampAll(CAMERAINFO_RGB_PATH)
depth_timestamps = timestampAll(CAMERAINFO_depth_PATH)

rgb_images = [cv2.imread(os.path.join(RAW_RGB_PATH, file)) for file in os.listdir(RAW_RGB_PATH)]
depth_images = [cv2.imread(os.path.join(RAW_depth_PATH, file), cv2.IMREAD_ANYDEPTH) for file in os.listdir(RAW_depth_PATH)]

# Synchronize images
synchronized_images = synchronize_images(rgb_timestamps, depth_timestamps, rgb_images, depth_images)

#---------------------match with each anchor image---------------------------------------------
anchors = [cv2.imread(os.path.join(ANCHOR_PATH, file)) for file in os.listdir(ANCHOR_PATH)]

def match_all_anchor(anchor_path, synchronized_images):
    for file in os.listdir(anchor_path):
        anchor = cv2.imread(os.path.join(anchor_path, file))    
        resized_anchor = cv2.resize(anchor, (round(0.1*anchor.shape[1]),round(0.1*anchor.shape[0])))
        scores = []    
        for (rgb, depth) in synchronized_images:
            scores.append(retrieve_objects(resized_anchor, rgb, depth))
        visualize_score(scores, file)

match_all_anchor(ANCHOR_PATH, synchronized_images)


