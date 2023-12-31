# This script performs the following tasks:

# 1. Feature Detection and Matching with LIGHTGLUE: Uses LightGlue for detecting and matching features between image pairs.
# 2. Feature Detection and Matching with ORB and Brute Force: Employs ORB for feature detection and brute force matching.
# 3. Saving Matched Keypoints: Saves the matched keypoints for each image into .txt files, separated by the matching method used.
# 4. Plotting Features and Matches: Visualizes features and matches for both LIGHTGLUE and ORB methods.
# 5. Saving Optical Flow Visualizations: Generates and saves optical flow visualizations for both LIGHTGLUE and ORB methods.
# 6. Setup and Initialization: Sets up necessary libraries and initializes devices, extractors, and matchers.
# 7. Image Path Configuration: Function to set and manage image paths for different sequences.


### SETUP ###
# Set the matplotlib backend to TkAgg
import matplotlib
matplotlib.use('TkAgg')

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch
import os

torch.set_grad_enabled(False)
#images = ("/home/anna/LightGlue/data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

# Initialize the LightGlue extractor and matcher
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=500) 

## ORB FAILED SEQUENCE 1 ##

def set_image_paths(sequence):
    paths = {
        1: ["/home/anna/LightGlue/orb_fail_data/firstfail/0165804999936.png",
            "/home/anna/LightGlue/orb_fail_data/firstfail/0165904999936.png"],
        2: ["/home/anna/LightGlue/orb_fail_data/firstfail/0165904999936.png",
            "/home/anna/LightGlue/orb_fail_data/firstfail/0166004999936.png"],
        3: ["/home/anna/LightGlue/orb_fail_data/secondfail/0300104999936.png",
            "/home/anna/LightGlue/orb_fail_data/secondfail/0300204999936.png"],
        4: ["/home/anna/LightGlue/orb_fail_data/secondfail/0300204999936.png",
            "/home/anna/LightGlue/orb_fail_data/secondfail/0300304999936.png"],
        5: ["/home/anna/LightGlue/orb_fail_data/thirdfail/0446004999936.png",
            "/home/anna/LightGlue/orb_fail_data/thirdfail/0446104999936.png"],
        6: ["/home/anna/LightGlue/orb_fail_data/thirdfail/0446104999936.png",
            "/home/anna/LightGlue/orb_fail_data/thirdfail/0446204999936.png"],
        # Distance Between Frames #
        7: ["/home/anna/LightGlue/orb_fail_data_distance/firstfail/0165804999936.png",
            "/home/anna/LightGlue/orb_fail_data_distance/firstfail/0166404999936.png"],
        8: ["/home/anna/LightGlue/orb_fail_data_distance/firstfail/0165804999936.png",
            "/home/anna/LightGlue/orb_fail_data_distance/firstfail/0167004999936.png"],
        9: ["/home/anna/LightGlue/orb_fail_data_distance/secondfail/0300104999936.png",
            "/home/anna/LightGlue/orb_fail_data_distance/secondfail/0300704999936.png"],
        10: ["/home/anna/LightGlue/orb_fail_data_distance/secondfail/0300104999936.png",
            "/home/anna/LightGlue/orb_fail_data_distance/secondfail/0301304999936.png"],
        11: ["/home/anna/LightGlue/orb_fail_data_distance/thirdfail/0446004999936.png",
            "/home/anna/LightGlue/orb_fail_data_distance/thirdfail/0446604999936.png"],
        12: ["/home/anna/LightGlue/orb_fail_data_distance/thirdfail/0446004999936.png",
            "/home/anna/LightGlue/orb_fail_data_distance/thirdfail/0447204999936.png"],
            #easy test
        13: ["/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0277504999936.png",
             "/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0277604999936.png"],
             #wierd results
        14: ["/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330504999936.png",
             "/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330604999936.png"],
        15: ["/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330604999936.png",
             "/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330704999936.png"],
        16: ["/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330704999936.png",
             "/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330804999936.png"],
        17: ["/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330804999936.png",
             "/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330904999936.png"],
        18: ["/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0330904999936.png",
             "/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0331004999936.png"],
        19: ["/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0331004999936.png",
             "/home/anna/testgithub2/UW-SLAM-preproject/Datasets/VAROS/cam0/data/0331104999936.png"],
    }

    if sequence not in paths:
        raise ValueError("Unknown sequence number.")
    return paths[sequence]

def lightglue(image0, image1):
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Annotations for number of features
    num_feats_image0 = len(kpts0)
    num_feats_image1 = len(kpts1)
    num_matches = len(matches)

    print(0, f'LG: Features in Image 1: {num_feats_image0}')
    print(1, f'LG: Features in Image 2: {num_feats_image1}')
    print(f'LG: Number of matches: {num_matches}')

    return matches01, kpts0, kpts1, m_kpts0, m_kpts1

def orb_bf(orb_image0, orb_image1):
    # Extract keypoints and descriptors with ORB
    orb_kpts0, orb_desc0 = orb.detectAndCompute(orb_image0, None)
    orb_kpts1, orb_desc1 = orb.detectAndCompute(orb_image1, None)

    # Stop if noe features where detected
    if len(orb_kpts0) == 0 or len(orb_desc0) == 0 or len(orb_kpts1) == 0 or len(orb_desc1) == 0:
        print("NO ORB FEATURES DETECTED")
        return False, False, False, False, False

    # Match ORB features with a BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb_matches = bf.match(orb_desc0, orb_desc1)

    # Sort the matches based on distance (best matches first)
    orb_matches = sorted(orb_matches, key=lambda x: x.distance)

    # Convert keypoints to a numpy array for plotting
    orb_kpts0 = np.array([kp.pt for kp in orb_kpts0], dtype=np.float32)
    orb_kpts1 = np.array([kp.pt for kp in orb_kpts1], dtype=np.float32)
    orb_matches = np.array([[m.queryIdx, m.trainIdx] for m in orb_matches])

    orb_m_kpts0 = orb_kpts0[orb_matches[:, 0]]
    orb_m_kpts1 = orb_kpts1[orb_matches[:, 1]]

    # Annotations for number of features
    orb_num_feats_image0 = len(orb_kpts0)
    orb_num_feats_image1 = len(orb_kpts1)
    orb_num_matches = len(orb_matches)

    print(0, f'ORB: Features in Image 1: {orb_num_feats_image0}')
    print(1, f'ORB: Features in Image 2: {orb_num_feats_image1}')
    print(f'ORB, BF: Number of matches: {orb_num_matches}')

    return orb_matches, orb_kpts0, orb_kpts1, orb_m_kpts0, orb_m_kpts1


def save_keypoints_to_file(sequence, keypoints, image_number, method='lightglue', output_dir='output'):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the file name with the output directory
    if method == 'lightglue':
        file_name = os.path.join(output_dir, f'test{sequence}_lightglue_image{image_number}.txt')
    else:  # Default to 'orb' if not 'lightglue'
        file_name = os.path.join(output_dir, f'test{sequence}_orb_image{image_number}.txt')

    with open(file_name, 'w') as file:
        if isinstance(keypoints, list):  # Assuming tensor data is presented as a list of tensors
            for tensor in keypoints:
                numbers = tensor.tolist() if hasattr(tensor, 'tolist') else tensor
                file.write(f'{numbers[0]} {numbers[1]}\n')
        else:  # Assuming NumPy array otherwise
            for point in keypoints:
                file.write(f'{point[0]} {point[1]}\n')

    print(f"Saved {method.upper()} matches for image {image_number} in sequence {sequence} to {file_name}")


def adjust_contrast(image):
    if isinstance(image, torch.Tensor):
        # Convert the tensor to a NumPy array
        image = image.cpu().numpy() if image.is_cuda else image.numpy()

    if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for channel-first format
        image = np.transpose(image, (1, 2, 0))  # Convert to channel-last format

    # Adjust the contrast (assuming image is in uint8 format)
    new_image = np.clip((image - image.min()) * (255 / (image.max() - image.min())), 128, 255)
    new_image = new_image.astype(np.uint8)
    
    return new_image


def plot_lightglue(image0, image1, kpts0, kpts1, m_kpts0, m_kpts1, matches01, sequence_number, mask, adjust_contrast_flag=False):
    contrast_suffix = ""
    if adjust_contrast_flag:
        image0 = adjust_contrast(image0)
        image1 = adjust_contrast(image1)
        contrast_suffix = "_lowcontrast"
     ## LIGHTGLUE MATCHES ##
    axes = viz2d.plot_images([image0, image1])
    # Plot all keypoints
    viz2d.plot_keypoints([kpts0, kpts1], colors=['yellow', 'yellow'], ps=10)

    if mask.any() == None:
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    else:    
        for i in range(len(m_kpts0)):
            color = 'red' if mask[i] == 0 else 'lime'
            viz2d.plot_matches(m_kpts0[i:i+1], m_kpts1[i:i+1], color=color, lw=0.2)
    
    #viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.add_text(0, "Lightglue" , fs=20)

    # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)

    # Save LIGHTGLUE matches figure
    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_lg_matches.eps', format='eps')
    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_lg_matches.pdf', format='pdf')

    ## LIGHTGLUE KEYPOINTS ##
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

    # Save LIGHTGLUE keypoints figure
    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_lg_features.eps', format='eps')
    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_lg_features.pdf', format='pdf')


def plot_orb_bf(image0, image1, orb_matches, orb_kpts0, orb_kpts1, sequence_number, mask, adjust_contrast_flag=False):
    contrast_suffix = ""
    if adjust_contrast_flag:
        image0 = adjust_contrast(image0)
        image1 = adjust_contrast(image1)
        contrast_suffix = "_lowcontrast"
    ## ORB BF MATCHES ##
    axes = viz2d.plot_images([image0, image1])
    # Plot all ORB keypoints
    viz2d.plot_keypoints([orb_kpts0, orb_kpts1], colors=['yellow', 'yellow'], ps=10)

    if mask.any() == None: 
        viz2d.plot_matches(orb_kpts0[orb_matches[:, 0]], orb_kpts1[orb_matches[:, 1]], color="lime", lw=0.2)
    else:
        correct_kpts0 = np.array([orb_kpts0[idx] for idx, m in zip(orb_matches[:, 0], mask) if m != 0])
        correct_kpts1 = np.array([orb_kpts1[idx] for idx, m in zip(orb_matches[:, 1], mask) if m != 0])
        if len(correct_kpts0) > 0 and len(correct_kpts1) > 0:
            viz2d.plot_matches(correct_kpts0, correct_kpts1, color="lime", lw=0.2)

        incorrect_kpts0 = np.array([orb_kpts0[idx] for idx, m in zip(orb_matches[:, 0], mask) if m == 0])
        incorrect_kpts1 = np.array([orb_kpts1[idx] for idx, m in zip(orb_matches[:, 1], mask) if m == 0])
        if len(incorrect_kpts0) > 0 and len(incorrect_kpts1) > 0:
            viz2d.plot_matches(incorrect_kpts0, incorrect_kpts1, color="red", lw=0.2)

    viz2d.add_text(0, "Orb & Brute Force" , fs=20)
    # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)

    # Save ORB BF matches figure
    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_orb_matches.eps', format='eps')
    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_orb_matches.pdf', format='pdf')

    ## ORB FEATURES ##
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([orb_kpts0, orb_kpts1], colors=['red', 'red'], ps=10)


    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_orb_features.eps', format='eps')
    plt.savefig(f'output/test{sequence_number}{contrast_suffix}_orb_features.pdf', format='pdf')

    #plt.show()


#Save optical flow visualization for a given method (LIGHTGLUE or ORB).
def save_optical_flow_visualization(image1_path, keypoints0, keypoints1, all_keypoints1, sequence_number, mask, method='lightglue', adjust_contrast_flag=False):

    ## Read and convert image
    image1_of = cv2.imread(image1_path, cv2.IMREAD_COLOR)

    contrast_suffix = ""
    if adjust_contrast_flag:
        image1_of = adjust_contrast(image1_of)
        contrast_suffix = "_lowcontrast"

    image1_rgb = cv2.cvtColor(image1_of, cv2.COLOR_BGR2RGB)


    # Get image dimensions for adaptive sizing
    height, width, _ = image1_rgb.shape
    aspect_ratio = width / height
    plt.figure(figsize=(aspect_ratio * 6, 6))

    # Display the second image
    plt.imshow(image1_rgb)

    # Convert keypoints1 to a set of tuples for easy comparison

    if isinstance(keypoints1, torch.Tensor):
        keypoints1 = keypoints1.cpu().numpy() if keypoints1.is_cuda else keypoints1.numpy()

    if isinstance(all_keypoints1, torch.Tensor):
        all_keypoints1 = all_keypoints1.cpu().numpy() if all_keypoints1.is_cuda else all_keypoints1.numpy()
    
    keypoints1_set = set(tuple(kp) for kp in keypoints1)

    print(len(keypoints1))
    print(len(all_keypoints1))
    # Filter all_keypoints1 to get only those not in keypoints1
    keypoints_not_matched = [kp for kp in all_keypoints1 if tuple(kp) not in keypoints1_set]
    print(len(keypoints_not_matched))
    # Convert keypoints1 to a list of cv2.KeyPoint objects
    keypoints_not_matched = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints_not_matched]

    # Draw keypoints on the second image
    image1_with_keypoints = cv2.drawKeypoints(image1_rgb, keypoints_not_matched, None, color=(0, 255, 255))

    # Choose colors based on method
    scatter_color, line_color = ('green', 'lime') if method == 'lightglue' else ('green', 'lime')

    #  Display the second image with keypoints using matplotlib
    plt.imshow(image1_with_keypoints)

    # Scatter keypoints on the second image,
    plt.scatter(keypoints1[:, 0], keypoints1[:, 1], color=scatter_color, marker='o', s=4, linewidths=0, alpha=1.0)

    if mask.any() == None:
        # Draw lines between corresponding features
        for (x1, y1), (x2, y2) in zip(keypoints0, keypoints1):
            plt.plot([x1, x2], [y1, y2], color='red', linewidth=1)

    else:
        # Draw lines between corresponding features with color based on mask
        for ((x1, y1), (x2, y2)), m in zip(zip(keypoints0, keypoints1), mask):
            color = 'red' if m == 0 else line_color  # Use red for outliers, else line color
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=1)

    plt.text(20, 20, f"Optical Flow, method: {method}", fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
    # Hide axis labels and ticks
    plt.axis('off')
    plt.tight_layout(pad=0.5)

    # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)

    # Save the figure
    plt.savefig(f'output/{method}_optical_flow_test{sequence_number}{contrast_suffix}.eps', format='eps')
    plt.savefig(f'output/{method}_optical_flow_test{sequence_number}{contrast_suffix}.pdf', format='pdf')

    #plt.show()


