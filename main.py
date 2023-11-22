from poseestimationdistance import *
from lightglue_orb import *
from findoutliers import *
from runonentireVAROS import *

"""
#Run LightGlue and ORb+BF on entire VAROS
"""
mean_rot_error_LG, mean_trans_error_LG, fails_LG, mean_rot_error_ORB, mean_trans_error_ORB, fails_ORB = runonentireVAROS()
print(mean_rot_error_LG)
print(mean_trans_error_LG)
print(fails_LG)
print(mean_rot_error_ORB)
print(mean_trans_error_ORB)
print(fails_ORB)




timestamps = np.array([
    [165804999936, 165904999936],
    [165904999936, 166004999936],
    [300104999936, 300204999936],
    [300204999936, 300304999936],
    [446004999936, 446104999936],
    [446104999936, 446204999936],
    #image pairs with longer baseline
    [165804999936, 166404999936],
    [165804999936, 167004999936],
    [300104999936, 300704999936],
    [300104999936, 301304999936],
    [446004999936, 446604999936],
    [446004999936, 447204999936],
    #easy example
    [277504999936, 277604999936]
])

## FIRST SAVE MATCHED POINTS AS IT IS NEEDED FOR POSE ESTIMATION ##
for sequence_number in range(13, len(timestamps)+1): 
    image0_path, image1_path = set_image_paths(sequence_number)
    print(image0_path)
    
    #Run LightGlue on test 1-12
    image0 = load_image(image0_path)
    image1 = load_image(image1_path)

    matches01, kpts0, kpts1, m_kpts0, m_kpts1 = lightglue(image0, image1)

    save_keypoints_to_file(sequence_number, m_kpts0, 1, method='lightglue', output_dir='output')
    save_keypoints_to_file(sequence_number, m_kpts1, 2, method='lightglue', output_dir='output')

    #Run ORB on test 1-12
    orb_image0 = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
    orb_image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    
    orb_matches, orb_kpts0, orb_kpts1, orb_m_kpts0, orb_m_kpts1 = orb_bf(orb_image0, orb_image1)
    
    save_keypoints_to_file(sequence_number, orb_m_kpts0, 1, method='orb', output_dir='output')

#Use files to calculate relative pose difference
save_keypoints_to_file(sequence_number, orb_m_kpts1, 2, method='orb', output_dir='output')

masksLG, masksORB = pose_estimation(timestamps)

## THEN USE MASK TO PLOT AND SAVE IMAGES ##
for sequence_number in range(1, len(timestamps)+1): 
    image0_path, image1_path = set_image_paths(sequence_number)
    print(image0_path)
    
    #Run LightGlue on test 1-12
    image0 = load_image(image0_path)
    image1 = load_image(image1_path)

    matches01, kpts0, kpts1, m_kpts0, m_kpts1 = lightglue(image0, image1)
    maskLG = np.array(masksLG[sequence_number-1])
    
    #Run ORB on test 1-12
    orb_image0 = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
    orb_image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    
    orb_matches, orb_kpts0, orb_kpts1, orb_m_kpts0, orb_m_kpts1 = orb_bf(orb_image0, orb_image1)
    maskORB = np.array(masksORB[sequence_number-1])

    #Plot results
    plot_lightglue(image0, image1, kpts0, kpts1, m_kpts0, m_kpts1, matches01, sequence_number, maskLG, adjust_contrast_flag=False)
    plot_orb_bf(image0, image1, orb_matches, orb_kpts0, orb_kpts1, sequence_number, maskORB, adjust_contrast_flag=False)
    save_optical_flow_visualization(image1_path, m_kpts0, m_kpts1, kpts1, sequence_number, maskLG, method='lightglue', adjust_contrast_flag=False)
    save_optical_flow_visualization(image1_path, orb_m_kpts0, orb_m_kpts1, orb_kpts1, sequence_number, maskORB, method='orb', adjust_contrast_flag=False)

pose_estimation(timestamps)



