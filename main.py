from poseestimationdistance import *
from lightglue_orb import *


#Run LightGlue on entire VAROS
    #In these functions there should be a call for a function that evaluates performance

#Run ORB + BruteForce on entire VAROS
    #In these functions there should be a call for a function that evaluates performance


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
    [446004999936, 447204999936]
])

for sequence_number in range(1, 13): 
    image0_path, image1_path = set_image_paths(sequence_number)
    
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
    save_keypoints_to_file(sequence_number, orb_m_kpts0, 2, method='orb', output_dir='output')
    
    #Plot results
    plot_lightglue(image0, image1, kpts0, kpts1, m_kpts0, m_kpts1, matches01)
    plot_orb_bf(image0, image1, orb_matches, orb_kpts0, orb_kpts1)
    save_optical_flow_visualization(image0_path, image1_path, m_kpts0, m_kpts1, sequence_number, method='lightglue')
    save_optical_flow_visualization(image0_path, image1_path, orb_m_kpts0, orb_m_kpts1, sequence_number, method='orb')
    

#Use files to calculate relative pose difference
pose_estimation(timestamps)

#Mark matches as correct or false based on GT rel pose
    #plot this


