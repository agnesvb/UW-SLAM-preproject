from poseestimationdistance import *


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

#Run LightGlue on test 1-12
    #Save results to file in this directory


#Run ORB on test 1-12
    #Save results to file in this directory


#Plot results 


#Use files to calculate relative pose difference
pose_estimation(timestamps)

#Mark matches as correct or false based on GT rel pose
    #plot this


