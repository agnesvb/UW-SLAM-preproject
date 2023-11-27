from poseestimationdistance import *
from lightglue_orb import *
import os

def get_pose(image0_path, image1_path, K, method):
    if method == "lightglue":
        image0 = load_image(image0_path)
        image1 = load_image(image1_path)
        matches01, kpts0, kpts1, m_kpts0, m_kpts1 = lightglue(image0, image1)
    elif method == "orb":
        image0 = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        matches01, kpts0, kpts1, m_kpts0, m_kpts1 = orb_bf(image0, image1)

    if np.all(matches01 == False):
        return 0
    m_kpts0 = np.array(m_kpts0)
    m_kpts1 = np.array(m_kpts1)
    E, mask = cv2.findEssentialMat(m_kpts0, m_kpts1, K,method=cv2.RANSAC, prob=0.999, threshold=1 )

    if np.shape(E) != (3,3):
        #Means we cannot recover pose
        print("number of matches: " + str(np.shape(m_kpts0)[0]))
        return 0
    
    points, R, t, mask = cv2.recoverPose(E,m_kpts0, m_kpts1) #The number of inliners which pass the cheirality test

    T_matrix = np.zeros((4,4))
    T_matrix[:3, :3] = R
    T_matrix[0:3,3] = t.T
    T_matrix[3,3] = 1

    return T_matrix

    

def runonentireVAROS():
    GT_path = "Datasets/VAROS/camM0_poses_transformation_matrix.csv"

    #K matrix
    fx = 990.323
    fy = 990.323 
    cx = 640.0
    cy = 360.0

    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])
   
    rot_errors_LG = []
    trans_errors_LG = []
    rot_errors_ORB = []
    trans_errors_ORB = []
    fails_LG = 0
    fails_ORB = 0 



    timestamp_path = "Datasets/VAROS/Timestamp_full.txt"
    with open(timestamp_path, "r") as file:
        # Read lines from the file and store them as an array of strings
        array_of_strings = file.readlines()

        # Remove leading and trailing whitespaces from each string
        array_of_strings = [line.strip() for line in array_of_strings]

    #runthrough VAROS
    counter = 0
    for value in array_of_strings[1:]:
        counter += 1 
        print("imagenr"+str(counter))
        if value == "0000104999936":
            image0_path = "Datasets/VAROS/cam0/data/0000004999936.png"
        else:
            image0_path = image1_path

        image1_path = "Datasets/VAROS/cam0/data/"+value+".png"
                
        


        #Get Ground truth pose
        if value == "0000104999936":
            t_start = 104999936
        else:
            t_start = t_end
        t_end = int(value)
        GT_data = read_GT_from_file(GT_path)
        GT_pose = relpose(GT_data, t_start, t_end)


        #For LightGlue
        T_LG = get_pose(image0_path, image1_path, K, "lightglue")
        if not np.all(T_LG == 0):
            rotation_error_deg, translation_error_deg = calculate_pose_error(T_LG, GT_pose)
            rot_errors_LG.append(rotation_error_deg)
            trans_errors_LG.append(translation_error_deg)
        else: 
            #saving fails as 360 degrees error for plotting
            fails_LG +=1
            rot_errors_LG.append(360)
            trans_errors_LG.append(360)
        #For ORB 
        T_ORB = get_pose(image0_path, image1_path, K, "orb")
        if not np.all(T_ORB == 0):
            rotation_error_deg, translation_error_deg = calculate_pose_error(T_ORB, GT_pose)
            rot_errors_ORB.append(rotation_error_deg)
            trans_errors_ORB.append(translation_error_deg)
        else: 
            #saving fails as 360 degrees error for plotting
            fails_ORB += 1
            rot_errors_ORB.append(360)
            trans_errors_ORB.append(360)


    
    #Removing fails from mean value
    mean_rot_error_LG = np.mean(rot_errors_LG[rot_errors_LG != 360])
    mean_trans_error_LG = np.mean(trans_errors_LG[trans_errors_LG != 360])
    mean_rot_error_ORB = np.mean(rot_errors_ORB[rot_errors_ORB != 360])
    mean_trans_error_ORB = np.mean(trans_errors_ORB[trans_errors_ORB != 360])
    with open("output/meanVAROS.txt", 'w') as file:
                file.write(f"{mean_rot_error_LG}\n")
                file.write(f"{mean_trans_error_LG}\n")
                file.write(f"{fails_LG}\n")
                file.write(f"{mean_rot_error_ORB}\n")
                file.write(f"{mean_trans_error_ORB}\n")
                file.write(f"{fails_ORB}\n")
    # Combine arrays into a 2D array
    combined_array = np.column_stack((rot_errors_LG, trans_errors_LG, rot_errors_ORB, trans_errors_ORB))

    # Specify the file path and write the array to a text file
    np.savetxt("output/EntireVAROS.txt", combined_array, fmt="%d", delimiter="\t")

    return mean_rot_error_LG, mean_trans_error_LG, fails_LG, mean_rot_error_ORB, mean_trans_error_ORB, fails_ORB