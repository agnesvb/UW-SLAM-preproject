import numpy as np
import cv2
import matplotlib.pyplot as plt

from poseestimationdistance import *

def outliersfromGT(pose,matches_paths,i):

    #K matrix
    fx = 990.323
    fy = 990.323 
    cx = 640.0
    cy = 360.0

    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

    invK= np.linalg.inv(K)

    R = pose[:3, :3]
    t = pose[0:3,3]
    E = np.cross(t,R)
    F = np.transpose(invK) @ E @ invK
    print(pose)
    print(E)

    #K matrix
    fx = 990.323
    fy = 990.323 
    cx = 640.0
    cy = 360.0

    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])
    
    matches1_path = matches_paths[i,0]
    matches2_path = matches_paths[i,1]
    
    coordinates1 = np.loadtxt(matches1_path)
    coordinates2 = np.loadtxt(matches2_path)
    print(i)
    for j in range(len(coordinates1)):
        if np.ndim(coordinates1) == 1:
            p1 = [coordinates1[0],coordinates1[1], 1]
            p2 = [coordinates2[0],coordinates2[1], 1]
        else:
            p1 = [coordinates1[j,0],coordinates1[j,1], 1]
            p2 = [coordinates2[j,0],coordinates2[j,1], 1]

        value = np.dot(np.transpose(p2),np.dot(E,p1))
        print(value)




def findoutliers(timestamps):
    GT_path = "Datasets/VAROS/camM0_poses_transformation_matrix.csv"

    matches_paths_LightGlue, matches_paths_ORB = get_paths()

    for i in range(np.shape(timestamps)[0]):

        t_start = timestamps[i,0]
        t_end = timestamps[i,1]
        print()
        #print("first picture is : " + str(t_start))
        #print("The second piscture is: " + str(t_end))
            
        print()
        #calculating ground truth pose
        GT_data = read_GT_from_file(GT_path)
        GT_pose = relpose(GT_data, t_start, t_end)
        #print(" The grond truth pose is: ")
        print(GT_pose)

        maskLG = outliersfromGT(GT_pose, matches_paths_LightGlue,i)
        maskORB = outliersfromGT(GT_pose, matches_paths_ORB, i)
        #print(maskLG)
    
    
