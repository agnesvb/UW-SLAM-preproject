import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_GT_from_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith("#"):  # Skip comments
                row = line.strip().split(",")
                timestamp = int(float(row[0]))
                R_matrix = np.zeros((3,3))
                t = np.zeros(3)
                T_matrix = np.zeros((4,4))
                R_matrix[0,0] = float(row[1])
                R_matrix[0,1] = float(row[2])
                R_matrix[0,2] = float(row[3])
                R_matrix[1,0] = float(row[5])
                R_matrix[1,1] = float(row[6])
                R_matrix[1,2] = float(row[7])
                R_matrix[2,0] = float(row[9])
                R_matrix[2,1] = float(row[10])
                R_matrix[2,2] = float(row[11])
                t[0] = float(row[4])
                t[1] = float(row[8])
                t[2] = float(row[12])
                T_matrix[:3, :3] = R_matrix
                T_matrix[0:3,3] = t.T
                T_matrix[3,3] = 1
                data[timestamp] = T_matrix
    return data

def relpose(data, t_start, t_end):
    T1 = data[t_start]
    T2 = data[t_end]

    T1inv= np.linalg.inv(T1)
    Trel = np.dot(T1inv,T2)

    return Trel

def relpose_from_matches(matches1_path, matches2_path, K):
    coordinates1 = np.loadtxt(matches1_path)
    coordinates2 = np.loadtxt(matches2_path)
    if np.shape(coordinates1)[0] < 5:
        print("ERROR: too few matches")
        return 0

    E, mask = cv2.findEssentialMat(coordinates1, coordinates2, K,method=cv2.RANSAC, prob=0.999, threshold=1 )

    if np.shape(E) != (3,3):
        #Means we cannot recover pose
        print("number of matches: " + str(np.shape(coordinates1)[0]))
        return 0
    
    points, R, t, mask = cv2.recoverPose(E,coordinates1, coordinates2) #The number of inliners which pass the cheirality test
    #R1, t1, R2, t2 = decompose_essential_matrix(E)
    #return R1, t1, R2, t2

    T_matrix = np.zeros((4,4))
    T_matrix[:3, :3] = R
    T_matrix[0:3,3] = t.T
    T_matrix[3,3] = 1

    return T_matrix

def decompose_essential_matrix(E):
    # Singular Value Decomposition of the essential matrix
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation matrices and translation vectors
    R1 = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U @ Vt)]]) @ Vt
    R2 = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -np.linalg.det(U @ Vt)]]) @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    # Ensure that the determinant of rotation matrices is positive
    if np.linalg.det(R1) < 0:
        R1 = -R1
        t1 = -t1

    if np.linalg.det(R2) < 0:
        R2 = -R2
        t2 = -t2

    return R1, t1, R2, t2

def calculate_pose_error(Matches_pose, GT_pose): 

    R_ab_Matches = Matches_pose[:3, :3]
    t_ab_Matches = Matches_pose[:3, 3]
    R_ab_GT = GT_pose[:3, :3]
    t_ab_GT = GT_pose[:3, 3]

    #Rotational error
    axisangle_Matches, _ = cv2.Rodrigues(R_ab_Matches)
    axisangle_GT, _ = cv2.Rodrigues(R_ab_GT)
    #the norm of the rotational vector is the angle of the axis angle representation
    angle_diff = np.linalg.norm(axisangle_Matches) - np.linalg.norm(axisangle_GT)
    # Convert to degrees
    rotation_error_deg = np.degrees(angle_diff)

    # ----ALTERNATIVE----
    R = R_ab_Matches - R_ab_GT
    R_angle_diff= np.arccos((np.trace(R)-1)/2)
    # Convert to degrees
    rotation_error_deg_alt2= np.degrees(R_angle_diff)
    #print("----")
    #print("the two alternatives for rotation error")
    #print(rotation_error_deg)
    #print(rotation_error_deg_alt2)
    #print("----")
    



    #Translational error
    normalized_t_ab_Matches = t_ab_Matches / np.linalg.norm(t_ab_Matches)
    normalized_t_ab_GT = t_ab_GT / np.linalg.norm(t_ab_GT)
    #Calculate the angular difference in radians
    dot_product = np.dot(normalized_t_ab_Matches, normalized_t_ab_GT)
    angular_difference_rad = np.arccos(dot_product)
    # Convert to degrees
    translation_error_deg = np.degrees(angular_difference_rad)

    if t_ab_Matches[2] < 0:
        print("negative z-value detected")
        t_ab_Matches_turned = - t_ab_Matches
        #Translational error
        normalized_t_ab_Matches_turned = t_ab_Matches_turned / np.linalg.norm(t_ab_Matches_turned)
        #Calculate the angular difference in radians
        dot_product_turned = np.dot(normalized_t_ab_Matches_turned, normalized_t_ab_GT)
        angular_difference_rad_turned = np.arccos(dot_product_turned)
        # Convert to degrees
        translation_error_deg_turned = np.degrees(angular_difference_rad_turned)
        print("the translational error for a turned t is:" + str(translation_error_deg_turned) + "degrees")



    return rotation_error_deg, translation_error_deg

def get_paths():
    matches_paths_LightGlue = np.array([
    ["output/test1_matches/test1_lightglue_image1.txt", "output/test1_matches/test1_lightglue_image2.txt"],
    ["output/test2_matches/test2_lightglue_image1.txt", "output/test2_matches/test2_lightglue_image2.txt"],
    ["output/test3_matches/test3_lightglue_image1.txt", "output/test3_matches/test3_lightglue_image2.txt"],
    ["output/test4_matches/test4_lightglue_image1.txt", "output/test4_matches/test4_lightglue_image2.txt"],
    ["output/test5_matches/test5_lightglue_image1.txt", "output/test5_matches/test5_lightglue_image2.txt"],
    ["output/test6_matches/test6_lightglue_image1.txt", "output/test6_matches/test6_lightglue_image2.txt"],
    #distance tests
    ["output/test7_matches/test7_lightglue_image1.txt","output/test7_matches/test7_lightglue_image2.txt"],
    ["output/test8_matches/test8_lightglue_image1.txt","output/test8_matches/test8_lightglue_image2.txt"],
    ["output/test9_matches/test9_lightglue_image1.txt","output/test9_matches/test9_lightglue_image2.txt"],
    ["output/test10_matches/test10_lightglue_image1.txt","output/test10_matches/test10_lightglue_image2.txt"],
    ["output/test11_matches/test11_lightglue_image1.txt","output/test11_matches/test11_lightglue_image2.txt"],
    ["output/test12_matches/test12_lightglue_image1.txt","output/test12_matches/test12_lightglue_image2.txt"]
])
    matches_paths_ORB = np.array([
    ["output/test1_matches/test1_orb_image1.txt", "output/test1_matches/test1_orb_image2.txt"],
    ["output/test2_matches/test2_orb_image1.txt", "output/test2_matches/test2_orb_image2.txt"],
    ["output/test3_matches/test3_orb_image1.txt", "output/test3_matches/test3_orb_image2.txt"],#THIS ONE FAILS AND IT IS SUPPOSED TO
    ["output/test4_matches/test4_orb_image1.txt", "output/test4_matches/test4_orb_image2.txt"],
    ["output/test5_matches/test5_orb_image1.txt", "output/test5_matches/test5_orb_image2.txt"],
    ["output/test6_matches/test6_orb_image1.txt", "output/test6_matches/test6_orb_image2.txt"],
    #distance tests
    ["output/test7_matches/test7_orb_image1.txt","output/test7_matches/test7_orb_image2.txt"],
    ["output/test8_matches/test8_orb_image1.txt","output/test8_matches/test8_orb_image2.txt"],
    ["output/test9_matches/test9_orb_image1.txt","output/test9_matches/test9_orb_image2.txt"],
    ["output/test10_matches/test10_orb_image1.txt","output/test10_matches/test10_orb_image2.txt"],
    ["output/test11_matches/test11_orb_image1.txt","output/test11_matches/test11_orb_image2.txt"],
    ["output/test12_matches/test12_orb_image1.txt","output/test12_matches/test12_orb_image2.txt"]
])

    return matches_paths_LightGlue, matches_paths_ORB



def pose_estimation(timestamps):
    #Calculating ground truth relative pose
    GT_path = "Datasets/VAROS/camM0_poses_transformation_matrix.csv"  
    
    #K matrix
    fx = 990.323
    fy = 990.323 
    cx = 640.0
    cy = 360.0

    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])
   

    
    #array of paths to matches for LightGlue, and ORB-features
    matches_paths_LightGlue, matches_paths_ORB = get_paths()

    #Saving results in a matrix, one row for each test and the first column for rotational error and second for translational error
    Results_LightGlue = np.zeros(np.shape(timestamps))
    Results_ORB = np.zeros(np.shape(timestamps))

    for i in range(np.shape(timestamps)[0]):
        print()
        print()
        print("Running test:" + str(i+1))

        t_start = timestamps[i,0]
        t_end = timestamps[i,1]
        print()
        print("first picture is : " + str(t_start))
        print("The second piscture is: " + str(t_end))
        
        print()
        #calculating ground truth pose
        GT_data = read_GT_from_file(GT_path)
        GT_pose = relpose(GT_data, t_start, t_end)
        print(" The grond truth pose is: ")
        print(GT_pose)
        print("the baseline is")
        baseline = np.linalg.norm(GT_pose[:3, 3])
        print(baseline)
        print()

        #First for LightGlue
        matches1_path = matches_paths_LightGlue[i,0]
        matches2_path = matches_paths_LightGlue[i,1]


        Matches_pose = relpose_from_matches(matches1_path, matches2_path, K)
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
        
        Matches_pose = relpose_from_matches(matches1_path, matches2_path, K)
        if not np.all(Matches_pose == 0):
            rotation_error_deg, translation_error_deg = calculate_pose_error(Matches_pose, GT_pose)
            print("And the pose from the ORB-features with brute force matches is:")
            print(Matches_pose)
            print(f"Rotation Error: {rotation_error_deg:.2f} degrees")
            print(f"Translation Error: {translation_error_deg:.2f} degrees")
        else: 
            print("Could not recover pose for ORB-features with brute force matches, probably to few matches")

    


    
    
    


    
    

    
      