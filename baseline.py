import numpy as np
import cv2
import matplotlib.pyplot as plt

file_path = "Datasets/VAROS/camM0_poses_transformation_matrix.csv"
data = {}
R = {}
t = {}
baselines = []
T_matrix = np.zeros((4,4))
with open(file_path, 'r') as file:
    for line in file:
        if not line.startswith("#"):  # Skip comments
            pastT = T_matrix
            row = line.strip().split(",")
            timestamp = int(float(row[0]))
            R_matrix = np.zeros((3,3))
            t_vec = np.zeros(3)
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
            t_vec[0] = float(row[4])
            t_vec[1] = float(row[8])
            t_vec[2] = float(row[12])
            T_matrix[:3, :3] = R_matrix
            T_matrix[0:3,3] = t_vec.T
            T_matrix[3,3] = 1
            data[timestamp] = T_matrix
            R[timestamp] = R_matrix
            t[timestamp] = t_vec
            if not np.all(pastT == np.zeros((4,4))):
                T1 = pastT
                T2 = T_matrix

                T1inv= np.linalg.inv(T1)
                Trel = np.dot(T1inv,T2)
                baselines.append(np.linalg.norm(Trel[:3, 3]))


print("the mean of the baseline is " + str(np.mean(baselines)) )