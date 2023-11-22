from lightglue_orb import *
from poseestimationdistance import *

#%%
def load_timestamps(file_path):
    # Read the file and convert each line to a float
    with open(file_path, 'r') as file:
        timestamps = np.array([float(line.strip()) for line in file])
    return timestamps

def set_image_paths_kitti(sequence):
    paths = {
            # KITTI
        1: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000000.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000001.png"],
        2: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000001.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000002.png"],
        3: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000002.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000003.png"],
        4: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000003.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000004.png"],
        5: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000004.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000005.png"],
        6: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000005.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000006.png"],
        7: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000006.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000007.png"],
        8: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000007.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000008.png"],
        9: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000008.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000009.png"],
        10: ["/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000009.png", 
             "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/image_0/000010.png"],
    }

    if sequence not in paths:
        raise ValueError("Unknown sequence number.")
    return paths[sequence]

def get_paths_kitti():
    matches_paths_lg_kitti = np.array([
    ["output/test13_lightglue_image1.txt", "output/test13_lightglue_image2.txt"],
    ["output/test14_lightglue_image1.txt", "output/test14_lightglue_image2.txt"],
    ["output/test15_lightglue_image1.txt", "output/test15_lightglue_image2.txt"],
    ["output/test16_lightglue_image1.txt", "output/test16_lightglue_image2.txt"],
    ["output/test17_lightglue_image1.txt", "output/test17_lightglue_image2.txt"],
    ["output/test18_lightglue_image1.txt", "output/test18_lightglue_image2.txt"],
    ["output/test19_lightglue_image1.txt", "output/test19_lightglue_image2.txt"],
    ["output/test20_lightglue_image1.txt", "output/test20_lightglue_image2.txt"],
    ["output/test21_lightglue_image1.txt", "output/test21_lightglue_image2.txt"],
    ["output/test22_lightglue_image1.txt", "output/test22_lightglue_image2.txt"]
])
    matches_paths_orb_kitti = np.array([
    ["output/test13_orb_image1.txt", "output/test13_orb_image2.txt"],
    ["output/test14_orb_image1.txt", "output/test14_orb_image2.txt"],
    ["output/test15_orb_image1.txt", "output/test15_orb_image2.txt"],
    ["output/test16_orb_image1.txt", "output/test16_orb_image2.txt"],
    ["output/test17_orb_image1.txt", "output/test17_orb_image2.txt"],
    ["output/test18_orb_image1.txt", "output/test18_orb_image2.txt"],
    ["output/test19_orb_image1.txt", "output/test19_orb_image2.txt"],
    ["output/test20_orb_image1.txt", "output/test20_orb_image2.txt"],
    ["output/test21_orb_image1.txt", "output/test21_orb_image2.txt"],
    ["output/test22_orb_image1.txt", "output/test22_orb_image2.txt"]
])
    return matches_paths_lg_kitti, matches_paths_orb_kitti

def read_GT_from_KITTI(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            if not line.startswith("#"):  # Skip comments
                values = line.strip().split()
                T_matrix = np.zeros((4,4))
                T_matrix[0, :] = [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
                T_matrix[1, :] = [float(values[4]), float(values[5]), float(values[6]), float(values[7])]
                T_matrix[2, :] = [float(values[8]), float(values[9]), float(values[10]), float(values[11])]
                T_matrix[3, :] = [0, 0, 0, 1]  # Homogeneous transformation matrix
                data[index] = T_matrix
    return data

def pose_estimation_kitti(timestamps):
    #Calculating ground truth relative pose
    GT_path = "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/poses/00.txt"
    GT_data = read_GT_from_KITTI(GT_path)  
    
    #K matrix
    fx = 718.856
    fy = 718.856 
    cx = 607.1928 
    cy = 185.2157

    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])
   

    
    #array of paths to matches for LightGlue, and ORB-features
    matches_paths_LightGlue, matches_paths_ORB = get_paths_kitti()

    #Saving results in a matrix, one row for each test and the first column for rotational error and second for translational error
    Results_LightGlue = np.zeros(np.shape(timestamps))
    Results_ORB = np.zeros(np.shape(timestamps))

    for i in range(len(timestamps) - 1):
        print("\nRunning test:", i + 1)

        # The ground truth pose for the current and next frame
        GT_pose_current = GT_data[i]
        GT_pose_next = GT_data[i + 1]

        # Calculate the relative pose
        GT_pose = relpose(GT_data, i, i + 1)
        print(" The grond truth pose is: ")
        print(GT_pose)
        print("the baseline is")
        baseline = np.linalg.norm(GT_pose[:3, 3])
        print(baseline)
        print()

        #First for LightGlue
        matches1_path = matches_paths_LightGlue[i,0]
        matches2_path = matches_paths_LightGlue[i,1]


        Matches_pose, mask_lg = relpose_from_matches(matches1_path, matches2_path, K)
        #print(mask_lg)
        if np.any(mask_lg == 255):
            print("OUTLIER LG")
        if not np.all(Matches_pose == 0):
            rotation_error_deg, translation_error_deg = calculate_pose_error(Matches_pose, GT_pose)
            print("And the pose from the LightGlue matches is:")
            print(Matches_pose)
            print(f"Rotation Error: {rotation_error_deg:.2f} degrees")
            print(f"Translation Error: {translation_error_deg:.2f} degrees")
        else:
            print("Could not recover pose for LightGlue matches, probably to few matches")

        print()
        #Then for ORB-features
        matches1_path = matches_paths_ORB[i,0]
        matches2_path = matches_paths_ORB[i,1]
        
        Matches_pose, mask_orb = relpose_from_matches(matches1_path, matches2_path, K)
        #print(mask_orb)
        if np.any(mask_orb == 255):
            print("OUTLIER orb")
        if not np.all(Matches_pose == 0):
            rotation_error_deg, translation_error_deg = calculate_pose_error(Matches_pose, GT_pose)
            print("And the pose from the ORB-features with brute force matches is:")
            print(Matches_pose)
            print(f"Rotation Error: {rotation_error_deg:.2f} degrees")
            print(f"Translation Error: {translation_error_deg:.2f} degrees")
        else: 
            print("Could not recover pose for ORB-features with brute force matches, probably to few matches")

    
    return mask_lg, mask_orb


# Path to the timestamps file
timestamps_file_path = "/home/anna/Documents/UW_SLAM/kitti_dataset/dataset/sequences/00/times.txt"
timestamps_kitti = load_timestamps(timestamps_file_path)
timestamps_kitti = timestamps_kitti[0:11]
print(timestamps_kitti[10])

mask_lg, mask_orb = pose_estimation_kitti(timestamps_kitti)

    
    


    
    

    
      
