import cv2
from stabilization_functions import *
from shutil import copyfile


output_vid = "stabilize.avi"


def stabilize(input_vid):

    # # ########################################### PART 2: Video Stabilization ##############################################
    #
    # # Choose parameters
    # WindowSize = 10  # Add your value here!
    # MaxIter = 2  # Add your value here!
    # NumLevels = 4  # Add your value here!
    # NumOfFrames = 10
    #
    # # Stabilize video
    # StabilizedVid = lucas_kanade_video_stabilization(input_vid, WindowSize, MaxIter, NumLevels, NumOfFrames)
    #

    copyfile(input_vid, output_vid)
    StabilizedVid = output_vid

    return StabilizedVid
