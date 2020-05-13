import cv2
import os
import sys

# this is a comment

def save_video_frames(input_vid_path, folder_name="input_frames"):

    # create the images folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # start the video capture
    input_vid = cv2.VideoCapture(input_vid_path)
    frame_num = 1

    while input_vid.isOpened():
        ret, frame = input_vid.read()
        if ret:
            # save frame to frames folder
            cv2.imwrite(f"{folder_name}/frame_{frame_num:03d}.jpg", frame)
            frame_num += 1
        else:
            break

    input_vid.release()
    cv2.destroyAllWindows()


def play_video(video_path):
    input_vid = cv2.VideoCapture(video_path)

    while input_vid.isOpened():
        ret, frame = input_vid.read()
        if ret:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    input_vid.release()
    cv2.destroyAllWindows()


# takes in an input video and saves an output video after processing
def write_video(input_vid_path, output_vid_path, processing_function):
    # TODO - change imgSize as necessary
    imgSize = (640, 360)
    frame_per_second = 30.0
    writer = cv2.VideoWriter(output_vid_path, cv2.VideoWriter_fourcc(*"MJPG"), frame_per_second, imgSize)

    cap = cv2.VideoCapture(input_vid_path)  # load the video
    while cap.isOpened():  # play the video by reading frame by frame
        ret, frame = cap.read()
        if ret == True:
            # image processing here
            # Our operations on the frame come here
            out_frame = processing_function(frame)
            writer.write(out_frame)  # save the frame into video file

            cv2.imshow('out_frame', out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        else:
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def create_video_from_image_folder(input_folder, output_file_path, shape):
    # TODO - change imgSize as necessary
    imgSize = shape
    frame_per_second = 30.0
    writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*"MJPG"), frame_per_second, imgSize)

    image_name_list = os.listdir(input_folder)
    for image_name in image_name_list:
        frame = cv2.imread(input_folder + os.sep + image_name)
        writer.write(frame)  # save the frame into video file
    writer.release()
    cv2.destroyAllWindows()


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()
